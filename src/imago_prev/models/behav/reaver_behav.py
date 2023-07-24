"""
Structures and routines for imitating Reaver policy and values
"""


import sys, os
import numpy as np
from functools import reduce
import operator

from enum import Enum
from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.transforms import ToTensor

from tqdm import tqdm
from absl import logging, flags, app
from sklearn import model_selection

from imago_prev.models.semframe.frames import *
from imago_prev.models.semframe.model_remapper import *
# from imago.models.semframe.model_gaussian import *

from imago_prev.models.semframe.trainer import *
from imago_prev.models.semframe.plot import *
import imago_prev.models.semframe.plot as splot

from imago_prev.models import model_util
import imago_prev.models.semframe.plot as splot

from sc2recorder.record_utils import load_episode_json, load_all_episodes
from matplotlib import pyplot as plt
import math
import pdb

from interestingness_xdrl.util.io import load_episodes
from interestingness_xdrl.util import io as xdrl_io


def _convert(idp):
    obs_dict = {}
    obs_dict['data'] = idp.observation.observation
    return obs_dict


def flatten_shape(shape):
    if isinstance(shape, list) or isinstance(shape, tuple):
        return int(prod(shape))
    else:
        return int(shape)

def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def scan(X):
    max_val = np.max(X)
    min_val = np.min(X)
    mean_val = np.mean(X)
    std_val = np.std(X)
    return "max={:.5f}, min={:.5f}, mean/std={:.5f}/{:.5f}".format(max_val, min_val, mean_val, std_val)


# Use existing VAE to encode and predict these targets
# In order Reaver action specs, with corresponding shapes
REAVER_ACT_SPECS_OLD = (("function_id", 16),
                ("screen", [16, 16]),
                ("minimap", [16, 16]),
                ("screen2", [16, 16]),
                ("queued", 2),
                ("control_group_act", 5),
                ("control_group_id", 10),
                ("select_add",  2),
                ("select_point_act", 4),
                ("select_worker", 4))

# This appears to be the updated one for Y1
# TODO: get this dynamically from the IDP
# This includes the action, the action_probs and action_factors (probabilities and named factor)

REAVER_ACT_SPECS = (
("act1", 4),
("act2", 4),
("act3", 4),
("act4", 4),
("act5", 24),
("act6", 24),
("act7", 24),
("act8", 24),
("act9", 6),
("act10", 6),
("act11", 6),
("act12", 6),
)

def idp2act_spec(idp):
    """ Given an InteractionDataPoint (from interestingness-xdrl), reads the action information
    and populates the act_specs"""
    act_spec = []
    for name, tensor in zip(idp.action_factors, idp.action_probs):
        act_spec.append((name, tensor.shape))
    return act_spec


def _mlp_block(num_in, num_out, dropout_rate=0.):
    """
    Sets up a MLP for translating the VAE encoder into the target representation
    """
    fc1_size = (max(num_in, num_out) + 1) // 2
    return nn.Sequential(
        nn.Linear(num_in, fc1_size),
        nn.Dropout(p=dropout_rate),
        nn.ReLU(),
        nn.Linear(fc1_size, num_out),
        nn.Dropout(p=dropout_rate),
        nn.Sigmoid()
    )


class Latent2Behavior(nn.Module):
    """ This piggybacks on top of the Remapper VAE, and sets up the targets for 
    """
    def __init__(self, vae_model, 
                 reaver_act_spec,
                 dropout_rate=0., deterministic=False, device="cpu"):
        super(Latent2Behavior, self).__init__()
        self.obs_model = vae_model
        self.device = device
        self.deterministic=deterministic
        self.SIZE_LATENT = self.obs_model.vae_model.N_LATENT
        self.act_spec_encoders = nn.ModuleDict() # Act spec encoders
        self.reaver_act_spec = reaver_act_spec
        
        # TODO: For spatial action encoders, use spatial network
        for name, act_spec_shape in self.reaver_act_spec:
            mlp_block = _mlp_block(self.SIZE_LATENT, flatten_shape(act_spec_shape))
            self.act_spec_encoders[name] = mlp_block
        self.act_spec_encoders = self.act_spec_encoders.to(self.device)
        self.value_encoder = nn.Sequential(
            nn.Linear(self.SIZE_LATENT, self.SIZE_LATENT // 2),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(self.SIZE_LATENT // 2, self.SIZE_LATENT // 4),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(self.SIZE_LATENT // 4, 1)
        ).to(self.device)
            
    def __str__(self):
        act_spec_str = "\n".join(["* {}\t{}".format(t[0], t[1]) for t in self.reaver_act_spec])
        return "Latent2Behavior, act_spec={}".format(act_spec_str)
            
    def forward(self, O):
        Xhats, Z_mu, Z_logvar, Z = self.obs_model(O, deterministic=self.deterministic)
        #Z = np.array([x['z'] for x in vae_res]) # Use deterministic for now
        #Z = ensure_torch(self.device, Z)
        
        # Assemble the results, but need to arrange them so they are returned instancewise
        Bhats_raw = []
        for name, mlp_fn in self.act_spec_encoders.items():
            bhat = mlp_fn(Z)
            bhat = bhat / torch.sum(bhat, dim=1).view(len(bhat), 1) # Normalize in place
            Bhats_raw.append(mlp_fn(Z))
        Bhats = Bhats_raw
        Vhat = self.value_encoder(Z)
        return Vhat, Bhats
    
    def forward_Z(self, Z):
        # Assemble the results, but need to arrange them so they are returned instancewise
        Bhats_raw = []
        for name, mlp_fn in self.act_spec_encoders.items():
            bhat = mlp_fn(Z)
            bhat = bhat / torch.sum(bhat, dim=1).view(len(bhat), 1) # Normalize in place
            Bhats_raw.append(mlp_fn(Z))
        Bhats = Bhats_raw
        Vhat = self.value_encoder(Z)
        return Vhat, Bhats    
    
    def convert2prob(self, Bhats):
        for idx in range(len(REAVER_ACT_SPECS)):
            name, act_spec_shape = REAVER_ACT_SPECS[idx]
            
OHAT = "Ohat"
Z_MU = "Z_mu"
Z_LOGVAR = "Z_logvar"
Z = "Z"
VHAT = "Vhat"
BHATS = "Bhats"
    
class ReaverBehavModel(nn.Module):
    def __init__(self, output_dir, components, 
                 device="cpu", 
                 n_hidden=128, beta=1e-6, n_latent=128,
                 deterministic_behav=False, lr=1e-3, height=144, width=192,
                 reaver_act_spec=REAVER_ACT_SPECS):
        super(ReaverBehavModel, self).__init__()
        self.model_dir = os.path.join(output_dir, "model")
        self.components = components
        self.device=device
        self.n_hidden, self.beta, self.n_latent = n_hidden, beta, n_latent
        self.obs_model = SC2VAE_remapper(self.components, device=device, n_hidden=n_hidden,
                                         n_latent=n_latent,
                                        height=height, width=width)
        self.behav_model = Latent2Behavior(self.obs_model, 
                                        reaver_act_spec=reaver_act_spec,
                                        device=device, deterministic=deterministic_behav)

        if self.model_save_exists():
            logging.info("Models exist, loading from {}".format(self.model_dir))
            step = self.load_model()
            logging.info(".. trained to step={}".format(step))
        else:
            logging.info("No save, found in {}, starting anew".format(self.model_dir))
    
    def __str__(self):
        ret = "ReaverBehavModel\n"
        ret += "Obs Model: {}".format(str(self.obs_model))
        ret += "Behav Model: {}".format(str(self.obs_model))
    
    def forward(self, datums):
        O = datums[0]
        Ohat, Z_mu, Z_logvar, Z = self.obs_model(O)
        Vhat, Bhats = self.behav_model(O)
        return {
            OHAT: Ohat,
            Z_MU: Z_mu,
            Z_LOGVAR: Z_logvar,
            Z: Z,
            VHAT: Vhat,
            BHATS: Bhats
        }
    
    def model_save_exists(self):
        return os.path.exists(os.path.join(self.model_dir, "recon_model.pt")) and  \
                os.path.exists(os.path.join(self.model_dir, "behav_model.pt"))

    def fwd_z2dict(self, z):
        recon_res = self.obs_model.fwd_z2dict(ensure_torch(self.device, z).view(1, -1))
    
    def save_model(self, step, loss):
        model_util.save_checkpoint(os.path.join(self.model_dir, "recon_model.pt"), self.obs_model, None, step, loss=loss)
        model_util.save_checkpoint(os.path.join(self.model_dir, "behav_model.pt"), self.behav_model, None, step, loss=loss)    

    def load_model(self, use_best=False):
        if use_best:
            logging.info("Loading best model")
            model_util.load_checkpoint(os.path.join(self.model_dir, "recon_model.best.pt"), self.obs_model, None, device=self.device)
            step = model_util.load_checkpoint(os.path.join(self.model_dir, "behav_model.best.pt"), self.behav_model, None, device=self.device)
        else:
            logging.info("Loading latest model")
            model_util.load_checkpoint(os.path.join(self.model_dir, "recon_model.pt"), self.obs_model, None, device=self.device)
            step = model_util.load_checkpoint(os.path.join(self.model_dir, "behav_model.pt"), self.behav_model, None, device=self.device)
        return step
    
    def sample_images(self, recon_dataloader, num_samples=5, to_tensor=True):
        if to_tensor:
            xforms = ToTensor()
        combined_imgs = []
        num_collected =0
        for datums in recon_dataloader:
            O = datums[0][0:num_samples]  # Limit in case batch size significantly larger
            Ohat, Z_mu, Z_logvar, Z = self.obs_model(O)
            obs_list = ohat2sc2(Ohat, self.components)
            combined_imgs.extend(splot.render_frames(obs_list, self.components))
            num_collected += len(O)
            if num_collected >= num_samples:
                break
        if to_tensor:
            return xforms(model_util.make_pil_grid(combined_imgs, padding=20, nrow=1))
        else:
            return model_util.make_pil_grid(combined_imgs, padding=20, nrow=1)
    
    
    