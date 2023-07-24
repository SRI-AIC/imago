import os
import io
import pdb

from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
import moviepy.editor as mp
from dataclasses import dataclass

from imago.domains import ImagoDomain, ANOM, ODIFF
from imago.perturb.cfs.gradient_based import realism_loss_gs
from imago.utils import *
from imago.analysis.anom.twostep_recon import TwoStepODiffAnomDet

"""
Routines for analyzing walks between latent points
"""

LINECOLOR_SCHED = ["b-", "r-", "c-", "m-", "y-"]

@dataclass
class CFPoint:
    anom_score:float
    odiff_score:float
    Z:np.ndarray


class ZTrajectory:
    """
    Captures the trajectories of variables, reconstructions along the latent trajectory from
    start_Z to end_Z, sampled at the number of steps.  Use compute_CF_points() and a perturbation
    direction to assemble counterfactual points for the desired perturbation.
    """
    def __init__(self, start_Z, end_Z,
                 domain:ImagoDomain,
                 steps=25):
        self.domain = domain
        self.total_steps = steps
        anom_model = TwoStepODiffAnomDet(domain.model, domain.odiff_fn)
        start_Z = ensure_torch(domain.model.device, start_Z)
        end_Z = ensure_torch(domain.model.device, end_Z)
        diff_Z = end_Z - start_Z
        self.start_Z, self.end_Z, self.diff_Z = start_Z, end_Z, diff_Z
        self.var_trajs = {}
        self.step_Ohats = []
        self.step_Thats = []
        start_Ohat = None

        # Populate the variable trajectories associated with each step, self.var_trajs
        for i in range(steps+1):
            # Collect stats on each step, including the endpoints
            step_Z = self._compute_step_Z(i)
            step_Ohat, step_That = domain.model.forward_Z(step_Z,
                                                   ret_for_training=False)
            self.step_Ohats.append(step_Ohat)
            self.step_Thats.append(step_That)
            if start_Ohat is None:
                # Cache the starting observation for computing the observation diff
                start_Ohat = step_Ohat
            # Collect the anomaly score and add to list trajectory vars
            anom_score, anom_label = anom_model.score(step_Z)
            odiff_score = domain.odiff_fn(start_Ohat, step_Ohat)
            step_That[ANOM] = anom_score
            step_That[ODIFF] = odiff_score
            for var_name, var_value in step_That.items():
                if var_name not in self.var_trajs:
                    self.var_trajs[var_name] = []
                self.var_trajs[var_name].append(ensure_item(var_value))
        assert (steps + 1) == len(list(self.var_trajs.values())[0])  # Sanity check

    def _compute_step_Z(self, step):
        """ Given the step number, returns the corresponding interpolated Z"""
        assert 0 <= step <= self.total_steps
        perc = step / self.total_steps
        step_Z = ensure_torch(self.domain.model.device, self.start_Z + perc * self.diff_Z)
        return step_Z

    def get_cf(self):
        """ Gets the pt1 counterfactual"""


    def get_values(self, idx, items):
        """
        Given the trajectory step and the list of items to retrieve,
        returns the in-order list of values at that time slice.
        """
        ret = []
        for item in items:
            ret.append(self.var_trajs[item][idx])
        return ret

    def bracket_target(self, var_name, tgt_var_value):
        """
        Given the target variable and the value, finds the
        x step that brackets/covers the target value.
        """
        tgt_var_traj = self.var_trajs[var_name]
        prev_var_val = tgt_var_traj[0]
        for i, var_value in enumerate(tgt_var_traj):
            if bracketed(prev_var_val, var_value,
                         tgt_var_value):
                return i
            prev_var_val = var_value
        return None

    def bracket_targets(self, var_targets):
        """ Given a list of (variable name, target value) pairs,
        returns the ith step in the trajectory where this criteria
        is met.  If not found, does not include it in the return dict."""
        ret = {}
        for var_name, tgt_value in var_targets:
            tgt_perc = self.bracket_target(var_name, tgt_value)
            if tgt_perc is not None:
                ret[var_name] = tgt_perc
        return ret

    def render(self,
               save_fpath,
               render_vars=[],  # The variables to display on the plot.  First is primary (left), rest are parasitic (right)
               target_pairs=[]):  # Vertical lines for each of the targeted render values values.
        """ Generates a movie with the target variables and var values"""
        step_imgs = []
        for i in range(self.total_steps + 1):
            step_Ohat = self.step_Ohats[i]
            step_That = self.step_Thats[i]
            step_img = self.domain.render_fn(step_Ohat, step_That)
            step_imgs.append(step_img)
        # Generate the variable tracking plots
        var_imgs = []
        plot_x = np.array([i / self.total_steps for i in range(self.total_steps + 1)])  # X axis are the percentage
        for i in range(self.total_steps + 1):
            # Render for each frame
            perc = i / self.total_steps
            fig, ax = plt.subplots(1, figsize=(6, 2))
            plt.subplots_adjust(right=0.75)  # What fraction to use for parasitic axes
            ax_offset = 1.01
            var2color = {}
            for ridx, render_var in enumerate(render_vars):
                traj_y = self.var_trajs[render_var]
                color = LINECOLOR_SCHED[ridx % len(LINECOLOR_SCHED)]
                if ridx == 0:
                    plot_ax = ax.plot(plot_x, traj_y, color, label=render_var)[0]
                    ax.set_ylabel(render_var)
                    color = plot_ax.get_color()
                    ax.yaxis.label.set_color(color)
                    var2color[render_var] = color
                else:
                    tracked_ax = ax.twinx()
                    tracked_ax.set_ylabel(render_var)
                    tracked_ax.spines.right.set_position(("axes", ax_offset))
                    ax_offset += .17
                    plot_ax = tracked_ax.plot(plot_x, traj_y, color, label=render_var)[0]
                    color = plot_ax.get_color()
                    var2color[render_var] = color
                    tracked_ax.yaxis.label.set_color(color)
            # Render target lines
            for target_var, tgt_offset in self.bracket_targets(target_pairs).items():
                color = var2color[target_var]
                ax.axvline(tgt_offset / self.total_steps, color=color, linestyle="--")
            # Render current step line
            ax.axvline(perc, color='red', label='Step', linestyle='--')
            plt.title("{} vs Perc.".format(render_vars[0]))
            ax.set_xlabel("Perc.")
            ax.set_ylabel("{}".format(render_vars[0]))
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plot_img = Image.open(buf)
            var_imgs.append(plot_img)
        return self._assemble_movie(save_fpath, step_imgs, var_imgs)

    def _assemble_movie(self,
                       save_fpath, top_imgs, bottom_imgs,
                       bgcolor=(0,0,0), time_per_frame=0.2):
        assert len(top_imgs) == len(bottom_imgs)
        combined_step_imgs = []
        # Creat the topbottom buffer
        buffer_img = PIL.Image.new('RGB', (top_imgs[0].width, 50), bgcolor)
        for top_img, bottom_img in zip(top_imgs, bottom_imgs):
            combined_step_imgs.append(tile_images_vert([top_img, bottom_img, buffer_img], buffer=10))
        gif_fpath = Path(save_fpath.parent, "{}.gif".format(save_fpath.stem))
        duration = self.total_steps * time_per_frame
        combined_step_imgs[0].save(gif_fpath, save_all=True,
                                   append_images=combined_step_imgs[1:],
                                   duration=duration, loop=0)
        clip = mp.VideoFileClip(str(gif_fpath))
        mpg_fpath = Path(save_fpath.parent, "{}.mp4".format(save_fpath.stem))
        clip.write_videofile(str(mpg_fpath))
        os.remove(gif_fpath)
        return mpg_fpath

    def compute_CF_points(self, perturb, anom_thresh=107,
                          use_realism_adjustment=False):
        """ Computes the counterfactual points from the trajectory, relative to the criteria
        introduced in the perturbation.  Returns three points,
        - First point where the counterfactual is met
        - First non-anomalous point after counterfactual is met
        - Final endpoint (full traversal)

        If use_realism_adjustment is set to True, then bumps the vectors at the end to
        move them towards the plausible direction.
        """
        if use_realism_adjustment:
            anom_model = TwoStepODiffAnomDet(self.domain.model, self.domain.odiff_fn)
        val_order = [ANOM, ODIFF]
        starting_value = self.var_trajs[perturb.var_name][0]
        var_met_step = self.bracket_target(perturb.var_name, perturb.get_target_value(starting_value))
        # Get measures at the given point
        if var_met_step is None:
            pt1 = None  # Cannot meet this goal
            pt2 = None
        else:
            values = self.get_values(var_met_step, val_order)
            if use_realism_adjustment is False:
                pt1 = CFPoint(anom_score=values[0],
                              odiff_score=values[1],
                              Z=ensure_numpy(self._compute_step_Z(var_met_step)))
            else:
                # Adjust the Z and compute the new anomaly and odiff
                Z1 = ensure_numpy(self._compute_step_Z(var_met_step))
                #update_rec = ensure_numpy(realism_loss(Z1, self.domain, update_norm_scale=2))
                update_rec = ensure_numpy(realism_loss_gs(Z1, self.domain))
                Z2 = Z1 + update_rec
                updated_anom_score = anom_model.score(Z2)
                Ohat1, _ = self.domain.model.forward_Z(Z1, ret_for_training=False)
                Ohat2, _ = self.domain.model.forward_Z(Z2, ret_for_training=False)
                updated_odiff = self.domain.odiff_fn(Ohat1, Ohat2)
                pt1 = CFPoint(anom_score = updated_anom_score,
                              odiff_score = updated_odiff,
                              Z=Z2)
            # Now compute the step where the anomaly score <= threshold
            pt2 = None
            for pt_idx in range(var_met_step, self.total_steps):
                pt_values = self.get_values(pt_idx, val_order)
                if pt_values[0] <= anom_thresh:
                    pt2 = CFPoint(anom_score=pt_values[0], odiff_score=pt_values[1],
                                  Z=self._compute_step_Z(pt_idx))
                    break
        # Get score at final point and assemble the results
        final_pt = self.get_values(-1, val_order)
        ptN = CFPoint(anom_score=final_pt[0], odiff_score=final_pt[1], Z=self.end_Z)
        return pt1, pt2, ptN