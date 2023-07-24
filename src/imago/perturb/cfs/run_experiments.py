from dataclasses import dataclass
import numpy as np

from imago.analysis.anom.twostep_recon import TwoStepODiffAnomDet
from imago.analysis.walk import ZTrajectory
from imago.domains import ImagoDomain, OBS
from imago.perturb.cfs import GRAD, GRAD_NOREAL, ODIFF, MIN_INST_ODIFF, ANOM, NUN, INTERP1, INTERP2, CF_MET, INTERP1_ADDREAL
from imago.perturb.cfs.gradient_based import GradPerturber
from imago.perturb.cfs.cbr_nun import CBR_NUN
from imago.utils import ensure_torch, ensure_numpy

@dataclass
class Result:
    domain:ImagoDomain
    inst_idx: int
    name: str
    start_Z: np.ndarray
    Z: np.ndarray
    odiff: float
    min_obsdiff: float
    anom_score: float
    cf_met: bool
    start_value: float
    end_value: float

    def render(self, save_fpath, steps=25, render_vars=[], target_pairs=[]) -> None:
        ztraj = ZTrajectory(self.start_Z, self.Z, self.domain, steps=steps)
        ztraj.render(save_fpath, render_vars=render_vars, target_pairs=target_pairs)
        

class CFExps:
    """
    All-in-one for performing the suite of perturbation experiments.  This accepts a
    domain, desired counterfactual perturbation, and the dataset to use as a basis for perturbations.

    For example,
        from imago.domains.cartpole.cartpole_setup import load_domain, load_data

        train_ds, test_ds = load_data()
        cp_domain = load_domain(device="cuda:0")
        varname = VALUE_FUNCTION
        tgt_mag = 2 * value_std
        direction = +1
        perturb = Perturb(varname, direction, tgt_mag=tgt_mag)
        cf_exp = CFExps(cp_domain, perturb, train_ds)
        cf_exp.auto_experiment()  # Runs the experiment using 100 good starting points
    """
    def __init__(self, domain, perturb, inst_ds):
        self.domain = domain
        self.perturb = perturb
        self.grad_perturber = GradPerturber(self.domain, self.perturb)
        self.anom_model = TwoStepODiffAnomDet(self.domain.model, self.domain.odiff_fn)
        self.inst_ds = inst_ds

    def assemble_results(self, inst_idx, name, Ohat_q, That_q, Z_query, Z_cf):
        # Given the query and the counterfactual Z, computes the necessary
        # scores.
        Ohat_cf, That_cf = self.domain.model.forward_Z(ensure_torch(self.domain.model.device, Z_cf))
        odiff = self.domain.odiff_fn(Ohat_q, Ohat_cf)
        min_obsdiff = self.domain.min_obsdist(Ohat_cf, self.inst_ds)
        var_name = self.perturb.var_name
        anom_score, _ = self.anom_model.score(Z_cf)
        cf_met = self.grad_perturber.perturb.target_met(That_q[var_name], That_cf[var_name]).item()
        return Result(self.domain, 
                      inst_idx, name, ensure_numpy(Z_query), 
                      ensure_numpy(Z_cf), odiff, min_obsdiff, 
                      anom_score, cf_met, That_q[var_name], That_cf[var_name])

    def assemble_cf_only(self, inst_idx, name, Ohat_q, That_q, Z_query, Z_cf):
        # Given the query and the counterfactual Z, packages just the counterfactual
        Ohat_cf, That_cf = self.domain.model.forward_Z(ensure_torch(self.domain.model.device, Z_cf))
        var_name = self.perturb.var_name
        cf_met = self.grad_perturber.perturb.target_met(That_q[var_name], That_cf[var_name]).item()
        anom_score, _ = self.anom_model.score(Z_cf)
        return Result(self.domain, 
                      inst_idx, name, ensure_numpy(Z_query), 
                      ensure_numpy(Z_cf), 0., 0., 
                      anom_score, cf_met, That_q[var_name], That_cf[var_name])

    def score_inst(self, inst_idx, outcome_norm_scale=5, update_norm_scale=1,
                   assemble_cf_only=False,
                   use_grad=True, use_nun=True, verbose=False, max_grad_iterations=1000):
        """ Given the instance idx to use, runs the whole slate of counterfactuals and measures
        - If a counterfactual could be generated (CF met)
        - The observational difference
        - The anomaly score
        """
        O_q = ensure_torch(self.domain.model.device, self.inst_ds[inst_idx][OBS]).unsqueeze(0)
        Ohat_q, _, _, Z, That_q = self.domain.model(O_q)

        results = {}
        if verbose:
            print(f"score_inst, grad={use_grad} ({max_grad_iterations} iterations), nun={use_nun}")
            
        res_fn = self.assemble_results
        if assemble_cf_only:
            res_fn = self.assemble_cf_only

        if use_grad:
            # Get the gradient version
            _, _, Z_grad, _ = self.grad_perturber.process(Z, max_iterations=max_grad_iterations,
                                                        outcome_norm_scale=outcome_norm_scale,
                                                        update_norm_scale=update_norm_scale,
                                                        use_anom_loss=True)
            results[GRAD] = res_fn(inst_idx, GRAD, Ohat_q, That_q, Z, Z_grad)

            # Gradient, but no realism adjustment
            _, _, Z_grad2, _ = self.grad_perturber.process(Z, max_iterations=max_grad_iterations,
                                                        outcome_norm_scale=outcome_norm_scale,
                                                        update_norm_scale=update_norm_scale,
                                                        use_anom_loss=False)
            results[GRAD_NOREAL] = res_fn(inst_idx, GRAD_NOREAL, Ohat_q, That_q, Z, Z_grad2)

        # Get the NUN and interps
        if use_nun:
            query_datum = self.inst_ds[inst_idx]
            cbr_nun = CBR_NUN(query_datum, self.perturb, inst_ds=self.inst_ds, domain=self.domain, search_stepsize=5,
                            stop_at_first=True)
            nun = cbr_nun.get_neighbor(0)
            if nun is None:
                results[NUN] = Result(self.domain, inst_idx, NUN, ensure_numpy(Z), np.array(0), np.nan, np.nan, np.nan, False, np.nan, np.nan)
            else:
                nun_Ohat, _, _, nun_Z, nun_That = self.domain.model(ensure_torch(self.domain.model.device,
                                                                                nun.datum[OBS]).unsqueeze(0))
                nun_anom_score, nun_anom_label = self.anom_model.score(nun_Z)
                nun_start_value = query_datum[self.grad_perturber.perturb.var_name]
                nun_end_value = nun.datum[self.grad_perturber.perturb.var_name]
                results[NUN] = Result(self.domain, inst_idx, NUN, 
                                    ensure_numpy(Z), nun_Z, nun.odiff, 0, nun_anom_score, True,
                                    nun_start_value, nun_end_value)
            ztraj0 = cbr_nun.get_ztraj(0)
            if ztraj0 is not None:
                pt1, pt2, ptN = ztraj0.compute_CF_points(self.grad_perturber.perturb,
                                                        use_realism_adjustment=False)
                pt1_addreal, _, _ = ztraj0.compute_CF_points(self.grad_perturber.perturb,
                                                            use_realism_adjustment=True)
            else:
                pt1, pt2, ptN, pt1_addreal = None, None, None, None
            for name, pt in [(INTERP1, pt1), (INTERP2, pt2), (INTERP1_ADDREAL, pt1_addreal)]:
                if pt is None:
                    results[name] = Result(self.domain, inst_idx, name, ensure_numpy(Z), np.array(0), np.nan, np.nan, np.nan, False, np.nan, np.nan)
                if pt is not None:
                    results[name] = res_fn(inst_idx, name, Ohat_q, That_q, Z, pt.Z)
        return results

    def auto_experiment(self, num_exps=100):
        """ If no schedule is given, identifies good starting points for perturbation and runs the experiment"""
        schedule = self.perturb.get_good_queries(self.insta_ds, LIMIT=num_exps)
        all_results = []
        for inst_idx in schedule:
            all_results.append(self.score_inst(inst_idx))
        return all_results


def norm_value(value):
    if isinstance(value, float):
        return value
    return value.item()


def res_group2dict(res_group):
    res = {}
    first_result = list(res_group.items())[0][1]
    res["inst_idx"] = first_result.inst_idx
    for groupname, res_entry in res_group.items():
        res["{}_{}".format(groupname, ODIFF)] = res_entry.odiff
        res["{}_{}".format(groupname, MIN_INST_ODIFF)] = res_entry.min_obsdiff
        res["{}_{}".format(groupname, ANOM)] = res_entry.anom_score
        res["{}_{}".format(groupname, CF_MET)] = res_entry.cf_met
        res["{}_start_value".format(groupname)] = norm_value(res_entry.start_value)
        res["{}_end_value".format(groupname)] = norm_value(res_entry.end_value)
    return res