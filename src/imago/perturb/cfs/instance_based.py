from dataclasses import dataclass
from matplotlib import pyplot as plt
from pathlib import Path
import io
from PIL import Image
from tqdm import tqdm
import numpy as np
import pdb

from imago.analysis.walk import ZTrajectory
from imago.domains import OBS, forward, RISKINESS

"""
Instance based counterfactual selection, pulled from
a given dataset.
"""


@dataclass
class Neighbor:
    inst_idx: int  # Instance offset, in original dataset order
    datum:dict  # Neighbor datum
    ep:str
    frame:int
    neighbor_idx: int = -1 # Neighbor offset (absolute, regardless of stride)
    odiff:float=0.


@dataclass
class ZTrajPt:
    ztraj: ZTrajectory
    neighbor: Neighbor
    tgt_var_value: float


class NeighborhoodCFScanner:
    def __init__(self, domain,
                 inst_ds,
                 cf_perturb,
                 limit=None,  # Upper bound on number of valid insts to return from closest
                 stride=5,  # Stride in instances to skip over, as many are similar
                 results_dir="results",
                 valid_inst_sel_fn=None # Given list of valid insts (in order), returns subset
                 ):
        self.domain = domain
        self.inst_ds = inst_ds
        self.results_dir = Path(results_dir)
        self.cf_perturb = cf_perturb
        self.limit = limit
        self.stride = stride
        self.valid_inst_sel_fn = valid_inst_sel_fn

    def scan_neighborhood(self, query_datum):
        """
        Scans the instances, computes their observation diff (odiff) with
        the query datum, and returns neighbors in ascending odiff order.
        This organizes scenes by their episode and frame number, obtained from
        the instance dataset.
        """
        nn_tuples = []
        for inst_idx in tqdm(range(0, len(self.inst_ds), self.stride)):
            datum = self.inst_ds[inst_idx]
            ep, fr = self.inst_ds.get_epfr(inst_idx)
            Op = datum[OBS]
            odiff = self.domain.odiff_fn(np.expand_dims(query_datum[OBS], 0), np.expand_dims(Op, 0))
            nn_tuples.append(
                Neighbor(inst_idx=inst_idx, datum=datum, ep=ep, frame=fr, odiff=odiff) )
        sorted_list = sorted(nn_tuples, key=lambda x: x.odiff)
        for nidx, datum in enumerate(sorted_list):
            datum.neighbor_idx = nidx
        return sorted_list

    def identify_valid_cfs(self, query_datum,
                neighborhood_list=None):
        """
        Given the query datum, scans the feature neighborhood to identify candidate
        counterfactuals.  Criteria for a good counterfactual lies in the direction and
        scale variables.  Returns the instances in order.  Optionally, the valid
        instance selector may go through and deactivate
        """
        if neighborhood_list is None:
            neighborhood_list = self.scan_neighborhood(query_datum)
        query_cf_label = query_datum[self.cf_perturb.var_name]
        valid_cf_insts = []
        for n in neighborhood_list:
            cf_label_delta = n.datum[self.cf_perturb.var_name] - query_cf_label
            if self.cf_perturb.direction < 0:
                if cf_label_delta < -self.cf_perturb.tgt_mag:
                    valid_cf_insts.append(n)
            elif self.cf_perturb.direction > 0:
                # Want higher values
                if cf_label_delta > self.cf_perturb.tgt_mag:
                    valid_cf_insts.append(n)
            if self.limit is not None and len(valid_cf_insts) >= self.limit:
                break
        if self.valid_inst_sel_fn is not None:
            valid_cf_insts = self.valid_inst_sel_fn(valid_cf_insts)
        return valid_cf_insts

    def get_trajs(self, query_datum,
                  neighborhood_list=None):
        """
        Given the query datum, scans the feature neighborhood to identify candidate
        counterfactuals and returns these for further analysis.
        """
        if neighborhood_list is None:
            neighborhood_list = self.scan_neighborhood(query_datum)
        valid_cf_insts = self.identify_valid_cfs(query_datum,
                                                 neighborhood_list=neighborhood_list)
        _, _, _, start_Z, start_That = forward(self.domain.model, query_datum)
        tgt_var_value = start_That[self.cf_perturb.var_name].item()
        if self.cf_perturb.direction < 0:
            tgt_var_value -= self.cf_perturb.tgt_mag
        elif self.cf_perturb.direction > 0:
            tgt_var_value += self.cf_perturb.tgt_mag
        else:
            raise Exception("Unsupported tgt mag type={}".format(self.cf_perturb.tgt_mag))
        ret_traj_pts = []  # trajectories with tracking information
        for n in valid_cf_insts:
            end_datum = n.datum
            _, _, _, end_Z, _ = forward(self.domain.model, end_datum)
            ztraj = ZTrajectory(start_Z, end_Z, self.domain)
            ret_traj_pts.append(ZTrajPt(ztraj, n, tgt_var_value))
        return ret_traj_pts

    def process(self, query_datum, 
                neighborhood_list=None,
                name=None,
                additional_vars=[]):
        """
        Given the query datum, scans the feature neighborhood to identify candidate
        counterfactuals and saves these out to the results directory, indexing them
        by their odiff-ordered neighbor index.  Similar to get_trajs, except this
        emits out renders and summarizing documents.
        """
        if neighborhood_list is None:
            neighborhood_list = self.scan_neighborhood(query_datum)
        valid_cf_insts = self.identify_valid_cfs(query_datum, 
                                                 neighborhood_list=neighborhood_list)
        _, _, _, start_Z, start_That = forward(self.domain.model, query_datum)
        tgt_var_value = start_That[self.cf_perturb.var_name].item()
        if self.cf_perturb.direction < 0:
            tgt_var_value -= self.cf_perturb.tgt_mag
        elif self.cf_perturb.direction > 0:
            tgt_var_value += self.cf_perturb.tgt_mag
        else:
            raise Exception("Unsupported tgt mag type={}".format(self.cf_perturb.tgt_mag))
        save_dir = Path(self.results_dir, "cf_inst", str(self.cf_perturb))
        save_dir.mkdir(exist_ok=True, parents=True)
        query_neighborhood_img = plot_neighborhood(neighborhood_list, name,
                                                   valid_pts=valid_cf_insts)
        summary_fpath = Path(save_dir, "{}_neighborhood.png".format(name))
        query_neighborhood_img.save(summary_fpath)

        saved_n_tuples = []
        for n in valid_cf_insts:
            end_datum = n.datum
            _, _, _, end_Z, _ = forward(self.domain.model, end_datum)

            save_fpath = Path(save_dir, "{}_n{}_i{}_walk.mp4".format(name, n.neighbor_idx,
                                                                     n.inst_idx))
            saved_n_tuples.append((save_fpath, n))
            render_vars = [self.cf_perturb.var_name] + additional_vars
            ztraj = ZTrajectory(start_Z, end_Z, self.domain)
            ztraj.render(save_fpath, render_vars,
                         target_pairs=[(self.cf_perturb.var_name, tgt_var_value)])
        # Construct summary HTML page
        movies_str = ""
        for saved_n in saved_n_tuples:
            movies_str += """
            <br/>
            <p><b>neighbor={neighbor} idx={idx}</b></p>
            <iframe src="{src}" width="640" height="480">
            </iframe>
            """.format(neighbor=saved_n[1].neighbor_idx,
                       idx=saved_n[1].inst_idx, src=saved_n[0].name)
        with open(Path(save_dir, "{}_summary.html".format(name)), 'w') as f:
            f.write("""<html>
            <body>
            <title>{title}</title>
            <img src="{summary_fpath}"/>
            <br/>
            {movies}
            </body>
            </html>""".format(
                title=name,
                summary_fpath=summary_fpath.name,
                movies=movies_str
            ))


def plot_neighborhood(sorted_nn_tuples, name,
                      valid_pts=None):
    """ Plots the output of scan_neighborhood.
     If valid_pts given, their positions are drawn
     in using vertical lines."""
    plt.clf()
    plot_x = np.arange(0, len(sorted_nn_tuples), 1)
    plot_y = [nn_t.odiff for nn_t in sorted_nn_tuples]
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.plot(plot_x, plot_y)
    if valid_pts:
        for valid_pt in valid_pts:
            ax.axvline(valid_pt.neighbor_idx, color='red',
                       linestyle='--',
                       label="i{}".format(valid_pt.inst_idx))
    plt.title("{}: ODiff neighborhood".format(name))
    plt.xlabel("Neighbor Number")
    plt.ylabel("ODiff")
    plt.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)


# Simple valid instance policies
def _valid_threepts(valid_cf_isnts):
    """
    Retain first, middle, last
    """
    if len(valid_cf_isnts) <= 3:
        return valid_cf_isnts
    midpt = len(valid_cf_isnts) // 2
    res = []
    for idx, ninst in enumerate(valid_cf_isnts):
        if idx == 0 or idx == midpt or idx == (len(valid_cf_isnts) - 1):
            res.append(ninst)
    assert len(res) == 3
    return res


# Simple valid instance policies
def _valid_fivepts(valid_cf_ints):
    """
    Collect the first and last valid instances,
    and then a uniform sampling in between
    """
    if len(valid_cf_ints) <= 5:
        return valid_cf_ints
    delta = len(valid_cf_ints) // 4
    res = [valid_cf_ints[0]]
    midpts = [nidx for nidx in range(0, len(valid_cf_ints), delta)]
    res.extend([valid_cf_ints[i] for i in midpts[1:4]])
    res.append(valid_cf_ints[-1])
    assert len(res) == 5
    return res


def _first_valid(valid_cf_insts):
    """
    Returns the first (closest) valid instance
    """
    assert len(valid_cf_insts) > 0
    return valid_cf_insts[0]


VALID_SEL_NAME2FN = {
    'first': _first_valid,
    'threepts': _valid_threepts,
    'fivepts': _valid_fivepts
}

# General entry point for selecting valid functions
def get_valid_fn(name):
    if name not in VALID_SEL_NAME2FN:
        raise Exception("Method={} not in valid selection functions".format(name))
    return VALID_SEL_NAME2FN[name]
