

from enum import Enum
from collections import namedtuple
import json
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

from imago_prev.models.model_util import save_checkpoint, load_checkpoint, checkpoint_exists
#from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

from absl import logging, flags
from glob import glob

import pdb

"""
Routines for converting each of the Semantic Frames in PySC2 into tensor form, along with
a dataset (SC2FrameDataset) for working with them.

To use the standard set used for Reaver, please see assemble_default_obs_components()
"""


class SC2FrameDataset:
    """ Looks like we can't actually return the NamedNumpyArrays themselves.  We instead return indices, 
    and use get_batch option on the list of idxes.  Currently, returning indices in the __getitem__ is
    the tricky way to return back types the collator in DataLoader can work with, as NamedNumpyArrays are
    currently not considered basic types.  For getting the frames themselves, look to get_batch()
    """
    def __init__(self, episodes_list):
        self.frames = []
        # Check to see if this is a list of episodes (list of lists).
        # Otherwise if not, it's a flat list of scenes
        if isinstance(episodes_list, list) and \
               isinstance(episodes_list[0], list):
            for episode in episodes_list:
                for frame in episode:
                    self.frames.append(frame)
        else:
            self.frames.extend(episodes_list)
    
    def __getitem__(self, idx):
        return idx
    
    def __len__(self):
        return len(self.frames)

    def get_batch(self, indices):
        return [self.frames[idx] for idx in indices]
        
        

class FrameType(Enum):
    categorical = 1
    numeric = 2

    
class CategoryLookup:
    """ Constructs a lookup that allows us to map between ID space and offset space.
    Used to allow us to only compute softmaxes only for IDs we care about.
    TODO: Serialization and deserialization."""
    def __init__(self, distinct_ids):
        self.distinct_ids = sorted(list(distinct_ids))
        self.uid2offset = {uid:offset for offset, uid in enumerate(self.distinct_ids) }
        self.offset2id = {offset:uid for uid, offset in self.uid2offset.items() }
    
    def convert2offset(self, Y):
        Yp = Y.copy()
        for uid, offset in self.uid2offset.items():
            Yp[Y == uid] = offset
        return Yp
    
    def convert2uid(self, Y):
        if isinstance(Y, torch.Tensor):
            Yp = Y.clone()
        else:
            Yp = Y.copy()
        for uid, offset in self.uid2offset.items():
            Yp[Y==offset] = uid
        return Yp
    
    def __str__(self):
        return "\n".join(["{}: {}".format(offset, uid) for uid, offset in self.uid2offset.items()])
    
    def save(self, fpath):
        with open(fpath, 'w') as f:
            for _, uid in self.offset2uid.items():
                f.write("{}\n".format(uid))
    
    @property
    def size(self):
        return len(self.uid2offset)
    
def make_category_lookup(sc2obs, name):
    distinct_uids = set()
    def _inner(observations):
        for obs in observations:
            semframe = obs['data']['feature_screen'][name]
            distinct_uids.update(np.unique(np.array(semframe)))
    # If list of lists (episodes), then go through each individually
    if isinstance(sc2obs, list):
        if isinstance(sc2obs[0], list):
            for observations in sc2obs:
                _inner(observations)
        else:
            _inner(sc2obs)
        return CategoryLookup([int(x) for x in distinct_uids]) # Need to convert uids into ints, as np.int64 cannot be value for Tensors
    raise Exception("Unknown input type!")
    
    
def make_category_lookup_arr(O, feature_idx):
    """
    Given an observation array sample and the type offset, finds 
    the unique samples values and assembles the category lookup for it
    """
    distinct_uids = np.unique(O[:, feature_idx, :, :])
    return CategoryLookup([int(x) for x in distinct_uids]) # Need to convert uids into ints, as np.int64 cannot be value for Tensors
    
# Expected components
# Will be also used to specify decode targets

class CategoricalFrame:
    """
    We are going to an embedding based model, where categorical elements will now have
    an embedding representation and the target value will be the 
    """
    def __init__(self, name, max_value, dim):
        assert isinstance(dim, int)
        self.name, self.max_value, self.dim = name, max_value, dim
        
    def __str__(self):
        ret = "CategoricalFrame name={}, max_value={}".format(self.name, self.max_value)
        return ret
    
    @property
    def size(self):
        return self.max_value
        
class NumericFrame:
    def __init__(self, name:str, max_value:int):
        self.name, self.max_value = name, max_value

    def __str__(self):
        return "{}:NumericFrame, max_value={}".format(self.name, self.max_value)
        
    def save(self, fpath):
        with open(fpath, 'w') as f:
            json.dump({"name": self.name, 
                       "max_value": self.max_value,
                       "type": "numeric" }, f)
            
class SparseNumericFrame:
    def __init__(self, name:str, max_value:int):
        self.name, self.max_value = name, max_value

    def __str__(self):
        return "{}:SparseNumericFrame, max_value={}".format(self.name, self.max_value)
        
    def save(self, fpath):
        with open(fpath, 'w') as f:
            json.dump({"name": self.name, 
                       "max_value": self.max_value,
                       "type": "numeric" }, f)            

def is_numeric_frame(frame):
    return isinstance(frame, NumericFrame) or isinstance(frame, SparseNumericFrame)
            
            
def assemble_default_obs_components(DIM=10):
    """
    Assembles the Y1+ set of SC2 observations, based upon items seen in the given episodes.
    Note that we are now filtering just to three types, and observations are now stored as 
    arrays indexing each.
    """
    COMPONENTS = [CategoricalFrame('player_relative', 5, DIM), 
                  CategoricalFrame('unit_type', 1000, DIM),
                  SparseNumericFrame('unit_hit_points_ratio', 255)]                 
#                  NumericFrame('unit_hit_points_ratio', 255)]
    action_prob_components = []
    return COMPONENTS


def logits2label(X, dim=0):
    """
    Given BxCxHxW array, converts into BxHxW where values are softmax offsets
    """
    if len(X.shape) == 2: # HxW, return as is
        return X
    if len(X.shape) == 3: # CxHxW
        return torch.argmax(X, dim=0)
    else:
        # BxCxHxW
        return torch.argmax(X, dim=1)


#
# Different observation storage formats
# - IDP, a single tensor shaped (batch, F, 64, 64), where F indexes the semantic feature channel. 
# - Ohat, the k/v outputs of forward inference, where each type of output is expaneded out in 
# shape (batch, channel, height, width).  
# - SC2, list of named dictionaries, where each dictionary describes a single scene.
#
    
def obs2sc2(O, components, device="cpu"):
    """ Given an observation tensor from IDP.observation, converts
    this into a SC2 NamedNumpyArray observation """
    ret_obs = {'data': { 'feature_screen': {} }} 
    for cidx, component in enumerate(components):
        X = ensure_numpy(O[cidx]).astype(int)
        ret_obs['data']['feature_screen'][component.name] = X
    return ret_obs
    
    
def ohat2sc2(Ohat, components, device="cpu"):
    """ Given a list of reconstruction outputs, corresponding to the components,
    stitches them back into the PySC2 list of per-scene NamedNumpyArrays."""
    batch_size = list(Ohat.values())[0].shape[0]
    ret_obs = [{'data': { 'feature_screen': {} }} for _ in range(batch_size)]
    for component in components:
        Oh = Ohat[component.name]
        for b_idx in range(batch_size):
            y = Oh[b_idx]
            is_numeric = is_numeric_frame(component)
            if not(is_numeric):
                y = logits2label(y) # Convert from offsets back into label/uid space
            else:
                # Rescale 0-1 back to maximum value
                y = y.squeeze(0)
                y *= component.max_value
                y = torch.round(y)
            y = ensure_numpy(y).astype(int) 
            ret_obs[b_idx]['data']['feature_screen'][component.name] = y
    return ret_obs


# Trial to convert Ohat back into the IDP tensor format
def ohat2idp(Ohat, components, device="cpu"):
    """ Given a list of reconstruction outputs, corresponding to the components,
    stitches them back into individual NamedNumpyArrays and returns them."""
    #Ohat = [ensure_torch(device, o) for o in Ohat]  # Ensure each of the component arrays are torch
    batch_size = list(Ohat.values())[0].shape[0]
    ret_stack = []
    for component in components:
        Oh = Ohat[component.name]
        for b_idx in range(batch_size):
            y = Oh[b_idx]
            is_numeric = isinstance(component, NumericFrame) or isinstance(component, SparseNumericFrame)
            if not(is_numeric):
                y = logits2label(y) # Convert from offsets back into label/uid space
            else:
                # Rescale 0-1 back to maximum value
                y = y.squeeze(0)
                y *= component.max_value
                y = torch.round(y)
            y = ensure_numpy(y).astype(int) 
            ret_stack.append(y)
    return np.stack(ret_stack)

    
# DEPRECATING    
#def xhat2obs(Xhat, components, device="cpu"):
#    """ Given a list of reconstruction outputs, corresponding to the components,
#    stitches them back into individual NamedNumpyArrays and returns them."""
#    Xhat = [ensure_torch(device, Xhat) for Xhat in Xhat]
#    batch_size = Xhat[0].shape[0]
#    ret_obs = [{'data': { 'feature_screen': {} }} for _ in range(batch_size)]
#    for Xh, component in zip(Xhat, components):
#        for b_idx in range(batch_size):
#            y = Xh[b_idx]
#            is_numeric = isinstance(component, NumericFrame)
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


def diff(F1, F2, component):
    """ Returns the difference between frames 1 and 2. """
    nz_fr1 = len(np.where(F1 != 0)[0])
    nz_fr2 = len(np.where(F2 != 0)[0])
    if isinstance(component, CategoricalFrame):
        err = len(np.where(F1 != F2)[0])
    elif isinstance(component, NumericFrame):
        diff = F1 - F2
        err = np.sqrt(np.power(diff, 2))
    return err, nz_fr1, nz_fr2 
    
    
    
