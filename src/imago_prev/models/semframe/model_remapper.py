
from enum import Enum
from collections import namedtuple
import torch
import torch.nn as nn
from sc2recorder.utils import ensure_numpy, ensure_torch, ensure_torch_long, set_random_seed

from enum import Enum
from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from sc2recorder.utils import ensure_numpy, ensure_torch, ensure_torch_long, set_random_seed

from imago_prev.models.model_util import load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from imago_prev.models.model_util import save_checkpoint, load_checkpoint, checkpoint_exists, make_pil_grid
#from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

from absl import logging, flags

from imago_prev import logits2label
from .frames import *
from .multiheads import *
from .plot import render_frames
#from torchvision.transforms import ToTensor



class ConvVAE_remapper(nn.Module):
    """
    Following the intuition of StyleGAN, adds an affine transformation
    """
    def __init__(self, NUM_INPUTS, N_TARGETS, N_LATENT=128, N_HIDDEN=32, height=64, width=64):
        super(ConvVAE_remapper, self).__init__()
        self.WIDTH, self.HEIGHT = height, width
        self.N_TARGETS = N_TARGETS
        self.N_LATENT, self.N_HIDDEN = N_LATENT, N_HIDDEN
        
        self.encoder = nn.Sequential(
            nn.Conv2d(NUM_INPUTS, N_HIDDEN, kernel_size=4, stride=2), # H = (31, 31)
            nn.ReLU(),
            nn.Conv2d(N_HIDDEN, N_HIDDEN * 2, kernel_size=4, stride=2), # H = (14, 14)
            nn.ReLU(),
            nn.Conv2d(N_HIDDEN * 2, N_HIDDEN * 4, kernel_size=4, stride=2), # H = (6, 6)
            nn.ReLU(),
            nn.Conv2d(N_HIDDEN * 4, N_HIDDEN * 8, kernel_size=4, stride=2), # H = (2, 2)
            nn.Sigmoid(),
            nn.Flatten()
        )
        
        # end channels multiplied by end area
        # TODO: Consider spatially aware latent
        h = N_HIDDEN * 8 * 7 * 10
        self.fc21 = nn.Linear(h, N_LATENT)  # mu
        self.fc22 = nn.Linear(h, N_LATENT)  # logvar
        
        self.mapper = nn.Sequential(
            nn.Linear(N_LATENT, N_LATENT),
            nn.ReLU(),
            nn.Linear(N_LATENT, N_LATENT),
            nn.ReLU(),
            nn.Linear(N_LATENT, N_LATENT),
            nn.ReLU()
        )
        
        self.aff = nn.Linear(N_LATENT, N_LATENT)

        #self.fc3  = nn.Linear(N_LATENT, h) # Decode first layer
        
        # 144 x 196 config
        self.pre_dec_shape=(6,9)
        
        self.fc3 = nn.Linear(N_LATENT, N_LATENT * self.pre_dec_shape[0] * self.pre_dec_shape[1]) # TODO: determine shape
        
        tconv1 = nn.ConvTranspose2d(N_LATENT, N_HIDDEN * 4, kernel_size=5, stride=2) # 5 x 5
        tconv2 = nn.ConvTranspose2d(N_HIDDEN * 4, N_HIDDEN * 2, kernel_size=5, stride=2) # 13 x 13
        tconv3 = nn.ConvTranspose2d(N_HIDDEN * 2, N_HIDDEN, kernel_size=6, stride=2) # 30 x 30
        tconv4 = nn.ConvTranspose2d(N_HIDDEN, N_TARGETS, kernel_size=6, stride=2) # 64 x 64
        self.decoder = nn.Sequential(
            # Unflatten2D(h, 1, 1), # Don't need this now 
            tconv1,
            nn.ReLU(),
            tconv2,
            nn.ReLU(),
            tconv3,
            nn.ReLU(),
            tconv4
        )
        
        # ACTIVATION LOGGING on deocoder using forward hooks
        self.last_decoder_activations = {}

        def setup_hook(name):
            def fwd_hook_fn(module, i, o):
                self.last_decoder_activations[name] = o.detach()
            return fwd_hook_fn

        tconv1.register_forward_hook(setup_hook('tconv1'))
        tconv2.register_forward_hook(setup_hook('tconv2'))
        tconv3.register_forward_hook(setup_hook('tconv3'))
        tconv4.register_forward_hook(setup_hook('tconv4'))
        
    def is_conditional(self):
        return True
        
    def encode(self, x):
        h1 = self.encoder(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps*std + mu
            return z
        else:
            return mu

    def decode(self, a):
        h3 = self.fc3(a)
        h3 = h3.view(-1, self.N_LATENT, self.pre_dec_shape[0], self.pre_dec_shape[1])
        return self.decoder(h3)

    def forward(self, x, deterministic=False):
        """ Similar to regular forward, but passes the z through an affine translation layer """
        mu, logvar = self.encode(x)
        if deterministic:
            z_pre = mu
        else:
            z_pre = self.reparameterize(mu, logvar)
        a = self.aff(self.mapper(z_pre))
        return self.decode(a), mu, logvar, z_pre, a
    
    def embed(self, x, reparameterize=False):
        mu, logvar = self.encode(x)
        if reparameterize:
            z_pre = self.reparameterize(mu, logvar)
            return self.aff(self.mapper(z_pre))
        else:
            return self.aff(self.mapper(mu))
        
    def map_affine(self, z_latent):
        return self.aff(self.mapper(z_latent))
    
    def forward_det(self, x):
        """ Deterministic, with no resampling (just use means)
        """
        mu, logvar = self.encode(x)
        z = mu
        return self.decode(a), mu, logvar, self.aff(self.mapper(z))


class SC2VAE_remapper(nn.Module):
    def __init__(self, components, targets_dim=10, device="cpu", n_latent=128, n_hidden=32,
                height=144, width=192):
        super(SC2VAE_remapper, self).__init__()
        self.device=device
        self.components = components # Cannot resort them now, these must match input order
        self.targets_dim = targets_dim
        self.sc2_mf_in = MultiFrameInput(self.components, device=device)
        self.sc2_mf_out = MultiFrameOutput(targets_dim, self.components, device=device)
        self.vae_model = ConvVAE_remapper(self.sc2_mf_in.dim(), targets_dim, N_LATENT=n_latent, N_HIDDEN=n_hidden, height=height, width=width).to(device)

    def __str__(self):
        return str(self)
        
    def forward(self, obs_batch, deterministic=False):
        x1 = self.sc2_mf_in(obs_batch)
        y1, z_mu, z_logvar, z, a = self.vae_model(x1, deterministic=deterministic)
        y2 = self.sc2_mf_out(y1)
        return y2, z_mu, z_logvar, z
    
    
    def forward_set_Z(self, Z):
        """ Follows forward, except allows Z to be set.  Returns the Y and A.
        Note: More efficient than forward 2 dictionary methods """
        batch_size = len(Z)
        A = self.vae_model.aff(self.vae_model.mapper(Z))
        Y = self.vae_model.decode(A)
        Y = self.sc2_mf_out(Y)
        return Y, A
    
    def forward_withA(self, obs_batch, deterministic=False):
        x1 = self.sc2_mf_in(obs_batch)
        y1, z_mu, z_logvar, z, a = self.vae_model(x1, deterministic=deterministic)
        y2 = self.sc2_mf_out(y1)
        return y2, z_mu, z_logvar, z, a
    
    def fwd2dict(self, obs_batch, deterministic=False):
        """ Returns a list of dictionaries, mimicking the SC2 NamedNumpyArray format,
        associated component, and z."""
        x1 = self.sc2_mf_in(obs_batch)
        y1, z_mu, z_logvar, z, a = self.vae_model(x1, deterministic=deterministic)
        y2 = self.sc2_mf_out(y1)
        
        ret = [{'z': ensure_numpy(z[i].squeeze(0)), 'z_mu': ensure_numpy(z_mu[i].squeeze(0)), 'z_logvar': ensure_numpy(z_logvar[i].squeeze(0)),
                'a': a,
                'data': {'feature_screen': {} }} 
               for i in range(len(obs_batch))]
        batch_size = len(obs_batch)
        for Y, component in zip(y2, self.components):
            for b_idx in range(len(obs_batch)):
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
                ret[b_idx]['data']['feature_screen'][component.name] = y
        return ret
        
    def fwd_z2dict(self, z):
        """ Returns a list of dictionaries, mimicking the SC2 NamedNumpyArray format,
        associated component, and z.  Note that this is the latent_z, not the affine remapper"""
        batch_size = len(z)
        a = self.vae_model.aff(self.vae_model.mapper(z))
        y1 = self.vae_model.decode(a)
        y2 = self.sc2_mf_out(y1)
        
        ret = [{'z': ensure_numpy(z[i].squeeze(0)), 
                'a': ensure_numpy(a[i].squeeze(0)),
                'data': {'feature_screen': {} }} 
               for i in range(batch_size)]     

        for Y, component in zip(y2, self.components):
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
                ret[b_idx]['data']['feature_screen'][component.name] = y
        return ret

    def fwd_a2dict(self, A):
        """ Returns a list of dictionaries, mimicking the SC2 NamedNumpyArray format,
        associated component, and A."""
        batch_size = len(A)
        y1 = self.vae_model.decode(A)
        y2 = self.sc2_mf_out(y1)
        
        ret = [{'a': ensure_numpy(A[i].squeeze(0)), 
                'data': {'feature_screen': {} }} 
               for i in range(batch_size)]     

        for Y, component in zip(y2, self.components):
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
                ret[b_idx]['data']['feature_screen'][component.name] = y
        return ret
    
    
    def y2obs(self, batch_Y):
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
    

def load_model(model_root, device="cpu", optimizer=None):
    """ Utility routine for loading this type of model.  Expects model to be organized as,
    model/
       components/*.json       # Semantic Frame components
       checkpoints/sc2vae.pt   # Checkpoint
    """
    components = assemble_default_obs_components()
    logging.info("Loading model components:\n{}".format("\n. . .\n".join([str(x) for x in components])))
    model = SC2VAE_remapper(components, device=device)
    load_checkpoint(os.path.join(model_root, "checkpoints/sc2vae.pt"), model, optimizer, device=device)
    return model