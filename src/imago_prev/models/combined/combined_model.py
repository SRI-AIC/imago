import pickle
import os, sys
import pathlib
from absl import logging
import numpy as np
from tqdm import tqdm
from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import math
import numpy as np
import pandas as pd
import PIL

import pdb
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

import itertools
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool

from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage import measure as ski_measure

import math 
import sklearn
import pickle

logging.set_verbosity(logging.INFO)



from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F

from imago_prev.models.behav.reaver_behav import REAVER_ACT_SPECS, scan, idp2act_spec, Latent2Behavior
from imago_prev.models.behav.losses import vae_loss, behavior_losses, JSD
from imago_prev.data.datasets import IDPDataset
from imago_prev.models.semframe.frames import make_category_lookup, CategoricalFrame, NumericFrame, SparseNumericFrame, assemble_default_obs_components, obs2sc2, ohat2sc2, ohat2idp, logits2label
from sc2recorder.utils import ensure_numpy, ensure_torch, ensure_torch_long, set_random_seed

import pprint

from interestingness_xdrl.util.io import load_object
from imago_prev.data.datasets import IDPDataset, ensure_torch_rec
from imago_prev.models.semframe.multiheads import MultiFrameInput, MultiFrameOutput

from imago_prev.models.behav.reaver_behav_trainer import *
from imago_prev.models.behav.rb_perturb import RBPerturbModel
import imago_prev.models.semframe.plot as splot
from imago_prev.models import model_util

from glob import glob

DEVICE="cuda:0"
from imago_prev.models.semframe import frames
DIM=10

# Learning parameters
LEARN_BEHAVIOR=True

# COMPONENTS = [CategoricalFrame('player_relative', 5, DIM)]
COMPONENTS = assemble_default_obs_components(DIM)
def tile_images(images, buffer=10):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths) + buffer*len(widths)
    max_height = max(heights)
    new_im = PIL.Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0] + buffer
    return new_im

def downsample_frame(frame):
    F2 = ski_measure.block_reduce(frame, (1, 2, 3), func=np.max) 
    F2a = F2[:, 4:68, :]  # Note: Strips away top 4 and bottom 4 rows
    return F2a

#def logits2label(X, dim=0):
#    """
#    Given BxCxHxW array, converts into BxHxW where values are softmax offsets
#    """
#    if len(X.shape) == 2: # HxW, return as is
#        return X
#    if len(X.shape) == 3: # CxHxW
#        return torch.argmax(X, dim=0)
#    else:
#        # BxCxHxW
#        return torch.argmax(X, dim=1)

#def obs2sc2(O, components, device="cpu"):
#    """ Given an observation tensor from IDP.observation, converts
#    this into a SC2 NamedNumpyArray observation """
#    ret_obs = {'data': { 'feature_screen': {} }} 
#    for cidx, component in enumerate(components):
#        X = ensure_numpy(O[cidx]).astype(int)
#        ret_obs['data']['feature_screen'][component.name] = X
#    return ret_obs
#
#def ohat2obs(Ohat, components, device="cpu"):
#    """ Given a list of reconstruction outputs, corresponding to the components,
#    stitches them back into individual NamedNumpyArrays and returns them."""
#    #Ohat = [ensure_torch(device, o) for o in Ohat]  # Ensure each of the component arrays are torch
#    batch_size = list(Ohat.values())[0].shape[0]
#    ret_obs = [{'data': { 'feature_screen': {} }} for _ in range(batch_size)]
#    for component in components:
#        Oh = Ohat[component.name]
#        for b_idx in range(batch_size):
#            y = Oh[b_idx]
#            is_numeric = isinstance(component, NumericFrame) or isinstance(component, SparseNumericFrame)
#            if not(is_numeric):
#                y = logits2label(y) # Convert from offsets back into label/uid space
#            else:
#                # Rescale 0-1 back to maximum value
#                y = y.squeeze(0)
#                y *= component.max_value
#                y = torch.round(y)
#            y = ensure_numpy(y).astype(int) 
#            ret_obs[b_idx]['data']['feature_screen'][component.name] = y
#    return ret_obs




class ConvZ(nn.Module):
    """ 
    Encoder component: Has the Convolutional2D portion, but also a latent output
    """
    def __init__(self, num_c_in, num_c_out, kernel_size=4, stride=2):
        super(ConvZ, self).__init__()
        self.num_c_in, self.num_c_out = num_c_in, num_c_out
        self.model = nn.Sequential(
                    nn.Conv2d(num_c_in, num_c_out, kernel_size=kernel_size, stride=stride), 
                    nn.BatchNorm2d(num_c_out),
                    nn.ReLU())
        
    def forward(self, X):
        return self.model(X)

    
class TranspConvZ(nn.Module):
    """ Transpose convolution.  Accepts 2 * latent_size inputs on decode"""
    def __init__(self, num_in, num_out,
                 kernel_size=4, stride=2):
        super(TranspConvZ, self).__init__()
        self.model = nn.Sequential(
                    nn.ConvTranspose2d(num_in, num_out, kernel_size=kernel_size, stride=stride),
                    nn.BatchNorm2d(num_out),
                    nn.ReLU())
        
    def forward(self, X):
        Y = self.model(X)
        return Y
    
    
class ConvVAE_remapper2(nn.Module):
    """
    Following the intuition of StyleGAN, adds an affine transformation.
    
    Here we use a dense latent approach, encoding at each conv layer and then 
    concatenating them into a single latent.
    """
    def __init__(self, NUM_INPUTS, N_TARGETS, 
                 N_LATENT=256, N_HIDDEN=32, height=64, width=64):
        super(ConvVAE_remapper2, self).__init__()
        self.WIDTH, self.HEIGHT = height, width
        self.N_TARGETS = N_TARGETS
        self.N_LATENT, self.N_HIDDEN = N_LATENT, N_HIDDEN
        
        self.encoding_schedule = [
            (NUM_INPUTS, N_HIDDEN, 4,2),
            (N_HIDDEN, N_HIDDEN, 4,2),
            (N_HIDDEN, N_HIDDEN, 4,2),
            (N_HIDDEN, N_HIDDEN, 4,2),            
        ]
        
        # Schedule for latents, encoding down to latents 
        # that will be concatenated
        self.enc_latent_schedule = ([1, 32, 31, 31],
                                    [1, 32, 14, 14],
                                    [1, 32, 6, 6])
        
        self.COMBINED_Z_DIM = 2 * N_LATENT
        
        self.encoder_list = nn.ModuleList(
            [ConvZ(c_in, c_out, k, s) for c_in, c_out, k, s in self.encoding_schedule]            
        )
        
        self.z_encoders = []
        for start_idx in range(0, len(self.enc_latent_schedule)):
            z_encoder = []
            for i in range(start_idx, len(self.enc_latent_schedule)):
                nc = self.enc_latent_schedule[i][1]
                z_encoder.append(ConvZ(nc, nc, kernel_size=4, stride=2))
            z_encoder = nn.Sequential(*z_encoder)
            self.z_encoders.append(z_encoder)
        self.z_encoders = nn.ModuleList(self.z_encoders)
        
        self.z2mu = nn.Linear(self.COMBINED_Z_DIM, N_LATENT)
        self.z2logvar = nn.Linear(self.COMBINED_Z_DIM, N_LATENT)        
        
        self.pre_dec_shape = (2,2)
        self.fc3 = nn.Linear(N_LATENT, N_LATENT * self.pre_dec_shape[0] * self.pre_dec_shape[1]) # TODO: determine shape

        self.decoder = nn.Sequential(
            TranspConvZ(N_LATENT, N_HIDDEN * 4, kernel_size=4, stride=2),
            TranspConvZ(N_HIDDEN * 4, N_HIDDEN * 2, kernel_size=4, stride=2),
            TranspConvZ(N_HIDDEN * 2, N_HIDDEN, kernel_size=4, stride=2),
            TranspConvZ(N_HIDDEN, N_TARGETS, kernel_size=6, stride=2),
        )
        
    def encode(self, X):
        layerwise_res = []
        layerwise_Z = []        
        batch_size = X.shape[0]
        
        for idx in range(len(self.encoder_list)):
            encoder_layer = self.encoder_list[idx]
            X = encoder_layer(X)
            layerwise_res.append(X)
            if idx < len(self.z_encoders):
                layer_Z = self.z_encoders[idx](X)
                layerwise_Z.append(layer_Z.view(batch_size, -1))
            else:
                layerwise_Z.append(X.view(batch_size, -1))
        combined_Z = torch.cat(layerwise_Z, 1)
        mu = self.z2mu(combined_Z)
        logvar = self.z2logvar(combined_Z)
        return mu, logvar

    def is_conditional(self):
        return True
        
#    def encode(self, x):
#        h1 = self.encoder(x)
#        return self.fc21(h1), self.fc22(h1)

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
        return self.decode(z_pre), mu, logvar, z_pre
    
    def forward_Z(self, z):
        """ Similar to regular forward, but passes the z through an affine translation layer """
        return self.decode(z)    
    
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



class SC2VAE_remapper2(nn.Module):
    def __init__(self, components, targets_dim=10, device="cpu", 
                height=64, width=64):
        super(SC2VAE_remapper2, self).__init__()
        self.device=device
        self.components = components
        self.targets_dim = targets_dim
        self.sc2_mf_in = MultiFrameInput(self.components, device=device)
        self.sc2_mf_out = MultiFrameOutput(targets_dim, self.components, device=device)
        self.vae_model = ConvVAE_remapper2(self.sc2_mf_in.dim(), targets_dim, height=height, width=width).to(device)

    def forward(self, obs_batch, deterministic=False):
        x1 = self.sc2_mf_in(obs_batch)
        y1, z_mu, z_logvar, z = self.vae_model(x1, deterministic=deterministic)
        y2 = self.sc2_mf_out(y1)
        return y2, z_mu, z_logvar, z
    
    
    def forward_Z(self, z):
        y1 = self.vae_model.forward_Z(z)
        y2 = self.sc2_mf_out(y1)
        return y2
    
    def forward_withA(self, obs_batch, deterministic=False):
        x1 = self.sc2_mf_in(obs_batch)
        y1, z_mu, z_logvar, z = self.vae_model(x1, deterministic=deterministic)
        y2 = self.sc2_mf_out(y1)
        return y2, z_mu, z_logvar, z, None
    
    
    def forward_set_Z(self, Z):
        """ Follows forward, except allows Z to be set.  Returns the Y and A.
        Note: More efficient than forward 2 dictionary methods """
        batch_size = len(Z)
        A = self.vae_model.aff(self.vae_model.mapper(Z))
        Y = self.vae_model.decode(A)
        Y = self.sc2_mf_out(Y)
        return Y, A

    
    def fwd2dict(self, obs_batch, deterministic=False):
        """ Returns a list of dictionaries, mimicking the SC2 NamedNumpyArray format,
        associated component, and z."""
        x1 = self.sc2_mf_in(obs_batch)
        y1, z_mu, z_logvar, z = self.vae_model(x1, deterministic=deterministic)
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

    
        
OHAT = "Ohat"
Z_MU = "Z_mu"
Z_LOGVAR = "Z_logvar"
Zvar = "Z"
VHAT = "Vhat"
BHATS = "Bhats"

class ReaverBehavModel2(nn.Module):
    def __init__(self, output_dir, components, 
                 device="cpu", 
                 deterministic_behav=False, lr=1e-3, height=64, width=64,
                 use_best=False,
                 reaver_act_spec=REAVER_ACT_SPECS):
        super(ReaverBehavModel2, self).__init__()
        self.model_dir = os.path.join(output_dir, "model")
        self.components = components
        self.device=device
        self.obs_model = SC2VAE_remapper2(self.components, device=device)
        self.behav_model = Latent2Behavior(self.obs_model, 
                                        reaver_act_spec=reaver_act_spec,
                                        device=device, deterministic=deterministic_behav)

        if self.model_save_exists():
            logging.info("Models exist, loading from {}".format(self.model_dir))
            step = self.load_model(use_best=use_best)
            logging.info(".. trained to step={}".format(step))
        else:
            logging.info("No save, found in {}, starting anew".format(self.model_dir))
    
    def forward(self, datums):
        O = datums[0]
        Ohat, Z_mu, Z_logvar, Z = self.obs_model(O)
        Vhat, Bhats = self.behav_model(O)
        return {
            OHAT: Ohat,
            Z_MU: Z_mu,
            Z_LOGVAR: Z_logvar,
            Zvar: Z,
            VHAT: Vhat,
            BHATS: Bhats
        }
    
    
    def forward_Z(self, Z):
        Ohat = self.obs_model.forward_Z(Z)
        Vhat, Bhats = self.behav_model.forward_Z(Z)
        return {
            OHAT: Ohat,
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
            O = datums[0][0:(num_samples - num_collected + 1)]
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
    