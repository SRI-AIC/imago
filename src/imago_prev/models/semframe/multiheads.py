"""
Implements the Semantic Frame ConvVAE multi-headed frame inputs and outputs
"""

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

from .frames import *
from .plot import render_frames
#from torchvision.transforms import ToTensor

import pdb

DIM=10
TARGETS_DIM = 8
#COMPONENTS = [CategoricalFrame('player_relative', 5, DIM), CategoricalFrame('unit_type', 255, DIM), \
#                  NumericFrame('unit_hit_points_ratio', 255)]

    
class MultiFrameInput(nn.Module):
    """ TODO: Move action probabilities into this framework """
    def __init__(self, components, device="cpu"):
        super(MultiFrameInput, self).__init__()
        self.device = device
        # Do not sort components, as we must follow specified order
        self.components = components
        self.embeddings = nn.ModuleDict()
        for component in self.components:
            if isinstance(component, CategoricalFrame):
                self.embeddings[component.name] = nn.Embedding(component.size, component.dim).to(device)
        
    def dim(self):
        """
        Computes the total output dimensions for all of the inputs stacked together
        """
        total = 0
        for component in self.components:
            if isinstance(component, NumericFrame) or isinstance(component, SparseNumericFrame):
                total += 1
            elif isinstance(component, CategoricalFrame):
                total += component.dim
        return total
        
    def forward(self, O):
        """
        This expects a Obs array (batch, feature_index, H, W), where the feature_index indxes the components.
        This stacks them up in the fol
    
        """
        O_p = []  # Updated observation, with categorical frames being expanded into their embedding representations
        for feat_idx, component in enumerate(self.components):
            if isinstance(component, NumericFrame) or isinstance(component, SparseNumericFrame):
                A = O[:,feat_idx,:,:].to(dtype=torch.float, device=self.device) # Convert into float for uniformity
                A /= component.max_value
                A = A.unsqueeze(-1) # Unpack the last layer, so we can stack with embeddings
                O_p.append(A)
            # Type number indexes the embedding to use
            elif isinstance(component, CategoricalFrame):
                A = self.embeddings[component.name]((O[:, feat_idx, :, :]).to(dtype=torch.long, device=self.device))
                O_p.append(A)
            else:
                raise TypeError("Unknown component type={}".format(component))
        O_p = torch.cat(O_p, 3).permute(0,3,1,2) # Move into BxCxHxW format, as embeddings tacked onto end
        return O_p

class GatedRegressionHead(nn.Module):
    def __init__(self, c_dim):
        super(GatedRegressionHead, self).__init__()
        self.c_dim = c_dim
        self.gate = nn.Sequential(nn.ConvTranspose2d(c_dim, 1, kernel_size=1, stride=1),
                                  nn.Sigmoid())
        self.guesser = nn.Sequential(nn.ConvTranspose2d(c_dim, 1, kernel_size=1, stride=1),
                                  nn.Sigmoid())
        # Because of the two stage loss, cache the results of the forward.
        # NOTE: This requires the loss be called immediately after the forward.
        self.last_gate = None
        self.last_guess = None
        
    def forward(self, X):
        H1 = self.gate(X)
        self.last_gate = H1
        M1 = H1 > 0.5
        H2 = self.guesser(X)
        self.last_guess = H2
        return M1 * H2

    
    
class MultiFrameOutput(nn.Module):
    """ MultiFrameOutput is implemented as a multi-head output, each with its own fully 
    connected layer. The output shape for each head is (batch_size, H, W, C), where C
    is the number of cells.  In the case of CategoricalFrames, this would be the
    number of target categories (as sigmoids), suitable for a NLL loss.
    This is to be paired with a decoder that generates a H x W grid with C channels.  These
    C channels will be passed to individual MLPs for finishing the inference.  
    
    Mecanically, a Flatten() operator is used first to convert into a linear H x W x C input
    to the MLP.
    """
    def __init__(self, c_dim, components, device="cpu"):
        super(MultiFrameOutput, self).__init__()
        self.device=device
        self.components = components # Must accept components in order
        self.c_dim = c_dim
        self.heads = nn.ModuleList()
        # Assemble sub-components
        for component in components:
            if isinstance(component, NumericFrame):
                head = nn.Sequential(nn.ConvTranspose2d(c_dim, 1, kernel_size=1, stride=1),
                                    nn.Sigmoid()).to(device)
            elif isinstance(component, SparseNumericFrame):
                head = GatedRegressionHead(c_dim).to(device)
            elif isinstance(component, CategoricalFrame):
                head = nn.Sequential(nn.ConvTranspose2d(c_dim, component.size,
                                                               kernel_size=1, stride=1),
                                    nn.ReLU()).to(device)
            self.heads.append(head)
            
    def __str__(self):
        ret = []
        for component, head in zip(self.components, self.heads):
            ret.append("{}: module={}".format(component.name, str(head).replace('\n', ', ')))
        return "\n\n".join(ret)
    
    def forward(self, X):
        """ Given the output of the decoder, a map dim (batch, H, W, C), performs forward inference on
        each of the heads and returns a list of outputs for each, where each dim is (batch, H x W x T),
        where T is dependent on the type (1 for numeric, |classes| for categorical).  Note that outputs here 
        are given in offsets (categorical) and 0-1 scaled (numeric) for loss functions.  Check
        fwd2dict in the full model for conversion into a SC2 namednumpyarray.
        
        Returns a dictionary where each component is indexed by the component's name.
        
        TODO: Stack or keep separate for pixel discriminator?"""
        outputs = {} # In form (B, 1) for numeric, (B, dim) for 
        for head, component in zip(self.heads, self.components):
            Y_head = head(X)
            outputs[component.name] = Y_head
        return outputs
        
    def loss_fn(self, X_guess, O_gold):
        """
        This has been adjusted to work with purely target arrays
        The O_gold are the gold observations, shape (B, C, H, W)
        The X_guess is a list of [(B, D1, H, W), (B, D2, H, W) ... (B, Dn, H, W)]
        where n is the number of components, and each D is the dimensionality used
        for the constituent loss function.
        """
        # Objectives work best compared by batch, so batch up each target tensor and
        # run the comparisons
        loss = 0.
        for cidx, (component, head) in enumerate(zip(self.components, self.heads)):
            Xhat = X_guess[component.name]
            if isinstance(component, CategoricalFrame):
                hyp = F.log_softmax(Xhat, dim=1)
                loss += F.nll_loss(hyp, O_gold[:,cidx,:,:].to(dtype=torch.long))
            elif isinstance(component, NumericFrame):
                target = O_gold[:,cidx,:,:].to(dtype=torch.float) / component.max_value
                loss += F.mse_loss(Xhat, target.squeeze(1))
            elif isinstance(component, SparseNumericFrame):
                if False:
                    # Use L1
                    target = O_gold[:,cidx,:,:].to(dtype=torch.float) / component.max_value
                    loss += F.mse_loss(Xhat, target.squeeze(1))
                    loss += torch.abs()
                if True:
                    # Do a two stage check.  First treat it as a 0-1 problem, then
                    # do the value.
                    H1 = head.last_gate
                    T1 = torch.zeros(O_gold[:,cidx,:,:].shape).to(self.device, dtype=torch.float)
                    T1[torch.where(O_gold[:,cidx,:,:] > 0)] = 1
                    loss1 = F.binary_cross_entropy(H1, T1)
                    loss += loss1

                    # 2nd stage, mask against 0/1 targets in stage 1
                    H2 = head.last_guess * T1
                    T2 = O_gold[:,cidx,:,:].to(dtype=torch.float) / component.max_value
                    loss2 = F.mse_loss(H2, T2.squeeze(1))
                    loss += loss2

        # TODO Add consistency loss
        
        return loss
                
        