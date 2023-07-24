""" Processes outputs from the imago.pysc2.observer recordings.

REPRESENTATION
Currently we are targeting a single one-hot target as the target representation.  Because
we can have multiple factions with the same kind of units, we reflect treat the incoming
SC2 unit_types as raw_unit_types, and add in allegiance offsets to represent allegiances
(self, ally, neutral, enemy).    See split_alleg_unitid() and merge_alleg_unit_ids().

This includes observational metadata, and the number of distinct units will be multiplied
by four, the total number of possible allegiances.

"""
import torch
from torch.utils.data import Dataset
import numpy as np
import itertools

from enum import IntEnum


class MetaFeat(IntEnum):
    FRIENDLY_COM_Y = 0  # Row
    FRIENDLY_COM_X = 1  # Column
    FRIENDLY_COM_STD_Y = 2
    FRIENDLY_COM_STD_X = 3
    BANDIT_COM_Y = 4
    BANDIT_COM_X = 5
    BANDIT_COM_STD_Y = 6
    BANDIT_COM_STD_X = 7
    #NEUTRAL_COM_Y = 8
    #NEUTRAL_COM_X = 9
    #NEUTRAL_COM_STD_Y = 10
    #NEUTRAL_COM_STD_X = 11
    #FRIENDLY_HIST_T48 = 12
    #BANDIT_HIST_T110 = 13


class XTypeEnum(IntEnum):
    UNIT_TYPE_ONEHOT = 0
    RGB_IMG = 1

#
# In order to use the one-hot representation with unit allegiances,
# we offset each unit id by (player_id - 1) * ALLEGIANCE_OFFSET
#
ALLEGIANCE_OFFSET = 1000

ALLEGIANCE_NAMES = ["None", "Blue", "Green", "White", "Red"]

def split_alleg_unitid(raw_unit_id):
    """ Convenience routine for splitting apart a raw unit ID
    into its player_id and unit_id components"""
    allegiance_idx = raw_unit_id // ALLEGIANCE_OFFSET
    unit_id = raw_unit_id % ALLEGIANCE_OFFSET
    return allegiance_idx, unit_id


def merge_alleg_unit_ids(allegiance_idx, unit_id):
    return (allegiance_idx * ALLEGIANCE_OFFSET) + unit_id


def get_obs_tensors(obs, x_type=XTypeEnum.UNIT_TYPE_ONEHOT):
    """ Converts the processed observation into tensors X, Y, C
    """
    if x_type == XTypeEnum.UNIT_TYPE_ONEHOT:
        X = obs['unit_type_onehot']
    else:
        img_obj = obs['data']['rgb_screen']
        img_obj = img_obj.resize((64,64))
        X = np.swapaxes(np.array(img_obj), 0, 2)
    Y = obs['unit_type_offset']
    C = obs['features_M']
    return X, Y, C


# Use this to capture interaction datapoint targets (see XDRL).  Use SC2FrameDataset for the reconstruction component
OBS = "observation"
VALUE = "value_function"
ACTION_PROBS = "action_probs_{}"

class IDPDataset(Dataset):
    """ Dataset for IDP datapoitns.
    If a list of lists (trajectories of IDPs) is given, flattens into a single list.
    """
    def __init__(self, idata_points):
        if len(idata_points) > 0 and isinstance(idata_points[0], list):
            # This is a list of lists, so flatten
            self.idata_points = list(itertools.chain(*idata_points))
        else:
            self.idata_points = idata_points  # Interaction Datapoints
        self.v_mean = 0
        self.v_std = 1
        
    def __len__(self):
        return len(self.idata_points)
    
    def __getitem__(self, idx):
        """ Returns the IDP components as a dictionary """
        idp = self.idata_points[idx]
        res = { OBS: torch.tensor(idp.observation), 
                VALUE: (torch.tensor(idp.value) - self.v_mean)/self.v_std }
        for idx, A in enumerate(idp.action_probs):
            k = ACTION_PROBS.format(idx)
            res[k] = torch.tensor(A)
        return res

    def get_single(self, idx):
        """ Returns the observation array (feature, h, w), value, and then the action probs in order.  Unsqueezes first dimension for singleton batch"""
        idp = self.idata_points[idx]
        res = { OBS: torch.tensor(idp.observation).unsqueeze(0), 
                VALUE: ((torch.tensor(idp.value) - self.v_mean)/self.v_std).unsqueeze(0) }
        for idx, A in enumerate(idp.action_probs):
            k = ACTION_PROBS.format(idx)
            res[k] = torch.tensor(A).unsqueeze(0)
        return res

    def __getitem_old__(self, idx):
        """ Returns the observation array (feature, h, w), value, and then the action probs in order"""
        idp = self.idata_points[idx]
        return [torch.tensor(idp.observation), 
                (torch.tensor(idp.value) - self.v_mean)/self.v_std] + \
                [torch.tensor(x) for x in idp.action_probs]
                
    
    def get_single_old(self, idx):
        """ Returns the observation array (feature, h, w), value, and then the action probs in order.  Unsqueezes first dimension for singleton batch"""
        idp = self.idata_points[idx]
        return [torch.tensor(idp.observation).unsqueeze(0), 
                ((torch.tensor(idp.value) - self.v_mean)/self.v_std).unsqueeze(0)] + \
                [torch.tensor(x).unsqueeze(0) for x in idp.action_probs]
    
    def compute_V_params(self, assign_self=True):
        """ Computes the mean and std parameters for the value function from data.
        Use set_V_params() to assign to this dataset.
        :param assign_self: Assign V params after computing"""
        Vs = [idp.value for idp in self.idata_points]
        v_mean = np.mean(Vs)
        v_std = np.std(Vs)
        if assign_self:
            self.set_V_params(v_mean, v_std)
        return v_mean, v_std
    
    def set_V_params(self, mean, std):
        """ Sets the mean and std for centering and standardizing the
        value estimates."""
        self.v_mean, self.v_std = mean, std
        
    def restore_V(self, v):
        return (v + self.v_mean) * self.v_std
    
def ensure_torch_rec(device, X):
    if isinstance(X, torch.Tensor):
        return X.to(device)
    elif isinstance(X, list):
        return [ensure_torch_rec(device, x) for x in X]
    else:
        raise TypeError("Unknown type={}".format(type(X)))
