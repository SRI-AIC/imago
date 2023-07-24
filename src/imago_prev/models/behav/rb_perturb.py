"""
Perturbation Model
"""
import os, pathlib
import tqdm
import numpy as np
from absl import logging, flags
import json
import torch
import torch.nn as nn

from imago_prev.models.semframe import frames
from sc2recorder.utils import ensure_numpy, ensure_torch, ensure_torch_long, set_random_seed
from imago_prev import logits2label
import imago_prev.models.semframe.plot as splot

from .reaver_behav import REAVER_ACT_SPECS, ReaverBehavModel, scan

FLAGS = flags.FLAGS

def _n2p(x):
    if isinstance(x, list):
        return [_n2p(y) for y in x]
    else:
        return ensure_numpy(x)

class RBPerturbModel(nn.Module):
    def __init__(self, rb_root, device="cpu"):
        super(RBPerturbModel, self).__init__()
        self.rb_root, self.device = rb_root, device
        #rb_model_dir = os.path.join(rb_root)
        components_dir = os.path.join(rb_root, "components")
        planes_fpath = os.path.join(rb_root, "perturb_planes.npz")
        meta_fpath = os.path.join(rb_root, "meta.json")
        if os.path.isfile(meta_fpath):
            with open(os.path.join(rb_root, "meta.json")) as f:
                meta_dict = json.load(f)
        else:
            meta_dict = {
                'V_SCALING_FACTOR': 30.280656814575195  # May need to retune this for new models
            }
        self.V_SCALING_FACTOR = meta_dict['V_SCALING_FACTOR']
        self.components = frames.assemble_default_obs_components()
        self.rb_model = ReaverBehavModel(rb_root, self.components, device=device)
        self.planes_by_label = {}
        for plane_label, plane_vec in np.load(planes_fpath).items():
            self.planes_by_label[plane_label] = plane_vec / np.linalg.norm(plane_vec)
            
    def __str__(self):
        ret = []
        ret.append("RBPerturb, model root={}, device={}".format(self.rb_root, self.device))
        ret.append("Perturb Planes #={}:".format(len(self.planes_by_label)))
        for plane_label, _ in self.planes_by_label.items():
            ret.append("\t{}".format(plane_label))
        return "\n".join(ret)
    
    def forward(self, O, deterministic=True, conv2numpy=False):
        Y, Z_mu, Z_logvar, Z, A = self.rb_model.vae_model.forward_withA(O,
                                                           deterministic=deterministic)
        bY, V = self.rb_model.behav_model.forward_Z(Z)
        
        dists = self._format_as_dists(bY)
        if conv2numpy:
            Y, Z_mu, Z_logvar, Z, A = _n2p(Y), ensure_numpy(Z_mu), ensure_numpy(Z_logvar), ensure_numpy(Z), ensure_numpy(A)
            bY, V = ensure_numpy(bY), ensure_numpy(V)
            
        return {
            "Y": Y,
            "Z_mu": Z_mu,
            "Z_logvar": Z_logvar,
            "Z": Z,
            "A": A,
            "bY": bY,
            "V": V * self.V_SCALING_FACTOR,
            "action_dists": dists
        }
    
    def forward_Z(self, Z, deterministic=True):
        Y, A = self.rb_model.vae_model.forward_set_Z(Z) # Get the observation
        bY, V = self.rb_model.behav_model.forward_Z(Z)  # Get behavior predictions
        
        dists = self._format_as_dists(bY)
        
        return {
            "Y": Y,
            "A": A,
            "bY": bY,
            "V": V * self.V_SCALING_FACTOR,
            "action_dists": dists
        }
    
    def perturb(self, obs, plane_label, alpha=1.0):
        """ Given a single observation, say from idp.observation, """
        p_vec = self.planes_by_label[plane_label]
        Y, Z_mu, Z_logvar, Z, A = self.rb_model.vae_model.forward_withA([obs],
                                                           deterministic=True)
        z = Z[0]
        z1 = ensure_torch(self.device, z0 + alpha * p_vec).view((1, -1))
        res_z1 = self.rb_model.vae_model.fwd_z2dict(z1)
        Y_hat_z1, V_hat_z1 = self.rb_model.behav_model.forward_Z(z1)
        value = ensure_numpy(V_hat_z1)[0][0]
        
        return_dists = {}
        for act_idx in range(len(REAVER_ACT_SPECS)):
            y_hat_z1 = Y_hat_z1[act_idx]
            act_spec = REAVER_ACT_SPECS[act_idx][0]
            
        return res_z1, value
    
    def _format_as_dists(self, behav_Y):
        """ Convenience routine for converting hypotheses into distributions.
        Normalizes the hypotheses and returns a dictionary indexed by the act_spec name."""
        ret = []
        for b_y in behav_Y:
            curr = {}
            for act_idx in range(len(REAVER_ACT_SPECS)):
                name, act_spec_shape = REAVER_ACT_SPECS[act_idx]    
                guess_dist = ensure_numpy(b_y[act_idx])
                guess_dist = guess_dist / np.sum(guess_dist)
                curr[name] = guess_dist
            ret.append(curr)
        return ret
    
        """ Given the Y tensor list, reformats and returns as a PySC2 observation.
        Currently the Y output returns it by observation primary, followed by batch
        indices.  This unpacks and returns by batch offset primary, with observations
        packed in each instance."""
        # batch_size = len(batch_Y)
        batch_Y = [ensure_torch(self.device, Y) for Y in batch_Y]
        batch_size = batch_Y[0].shape[0]
        ret_obs = [{'data': { 'feature_screen': {} }} for _ in range(batch_size)]
        for Y, component in zip(batch_Y, self.components):
            for b_idx in range(batch_size):
                y = Y[b_idx]
                is_numeric = isinstance(component, NumericFrame)
                if not(is_numeric):
                    y = component.convert2uid(logits2label(y)) # Convert from offsets back into label/uid space
                else:
                    # Rescale 0-1 back to maximum value
                    y = y.squeeze(0)
                    y *= component.max_value
                    y = torch.round(y)
                y = ensure_numpy(y).astype(int) 
                ret_obs[b_idx]['data']['feature_screen'][component.name] = y
        return ret_obs
    
    def display_Y(self, Y):
        """
        Convenience routine for displaying observation output (Y) as summary images
        """
        return splot.render_frames(self.rb_model.rb_model.vae_model.y2obs(Y), self.components)