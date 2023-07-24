"""
Environment specific datasets, and Dataset classes for handling
multiple target variables.
"""
import collections
import numpy as np
import torch
from torch.utils.data import Dataset

from imago.utils import ensure_numpy, ensure_torch, stats

OBS = "observation"
VALUE_FUNCTION = 'value_function'
ACTION_PROBS = "action_probs_{}"
CONFIDENCE = "Confidence"
GOAL_COND = "Goal Conduciveness"
RISKINESS = "Riskiness"
INCONGRUITY = "Incongruity"

ANOM = "Anomaly"
ODIFF = "ObsDiff"

var_shortname = {
    VALUE_FUNCTION: "value",
    CONFIDENCE: "conf",
    GOAL_COND : "goalcond",
    RISKINESS : "risk",
    INCONGRUITY: "incog"
}

class ImagoDomain:
    def __init__(self, model, odiff_fn, render_fn):
        self.model, self.odiff_fn, self.render_fn = model, odiff_fn, render_fn

    def min_obsdist(self, O, inst_ds, stride=5):
        """ Given an observation and a instance dataset, generates the minimal observational
        difference value between the observation and the dataset.  Uses a skip to avoid near-redundant
        frames."""
        obs_diffs = []
        for idx in range(0, len(inst_ds), stride):
            datum = inst_ds[idx]
            odiff = self.odiff_fn(bformat_obs(datum), O)
            obs_diffs.append(odiff)
        return min(obs_diffs)

class DictTensorDataset(Dataset):
    def __init__(self, O, targets_dict,
                 epfr_lookup=None # If present, DataFrame marking dataset offset to ep and frame
                 ):
        self.O = O  # Observations
        # Prefix (1, 1) fake spatial dimensions if needed
        # Collect up all the targets, doing a length equality sanity check
        self.T = collections.OrderedDict()
        for tgt_name, tgt_M in targets_dict.items():
            assert len(self.O) == len(tgt_M)
            assert tgt_name != OBS  # Ensure no collisions
            self.T[tgt_name] = tgt_M
        if epfr_lookup is not None:
            self.epfr_lookup = epfr_lookup
            assert len(epfr_lookup) == len(O)

    def __len__(self):
        return len(self.O)

    def __getitem__(self, idx):
        ret = collections.OrderedDict()
        O = self.O[idx]
        ret[OBS] = O
        for tgt_name, tgt_M in self.T.items():
            ret[tgt_name] = tgt_M[idx]
        return ret

    def contains_epfr(self, ep, fr):
        df = self.epfr_lookup
        idx = df[(df.episode == ep) & (df.frame == fr)].index
        return len(idx) > 0

    def get(self, ep, fr):
        df = self.epfr_lookup
        idx = df[(df.episode == ep) & (df.frame == fr)].index
        if len(idx) == 0:
            raise Exception("Ep and frame not found!")
        idx = idx[0]
        return self.__getitem__(idx)

    def get_epfr_idx(self, ep, fr):
        df = self.epfr_lookup
        idx = df[(df.episode == ep) & (df.frame == fr)].index
        if len(idx) == 0:
            raise Exception("Ep and frame not found!")
        idx = idx[0]
        return idx

    def get_epfr(self, idx):
        """ Given an idx offset, returns the corresponding
        episode and frame"""
        row = self.epfr_lookup.iloc[idx]
        return row.episode, row.frame

    def get_ep_idxes(self, ep):
        df = self.epfr_lookup
        idxes = df[df.episode == ep].index
        return idxes

    def get_eps(self):
        return np.unique(self.epfr_lookup.episode)

    def __str__(self):
        ret = "DictTensorDataset, size={}\n".format(len(self))
        ret += "Obs shape={}\n".format(self.O.shape)
        for tgt_name, tgt_M in self.T.items():
            ret += "\t{}:\t{}\n".format(tgt_name, stats(tgt_M))
        return ret

    def arange(self, start_idx, end_idx, step=1):
        """ Returns a subset of current as a new DictTensorDataset"""
        Op = self.O[start_idx:end_idx:step]
        new_T = collections.OrderedDict()
        for tgt_name, tgt_M in self.T.items():
            new_T[tgt_name] = tgt_M[start_idx:end_idx:step]
        return DictTensorDataset(Op, new_T)

    def save(self, output_fpath):
        M = {}
        M[OBS] = self.O
        M.update(self.T)
        np.savez(output_fpath, **M)

def load_from_npz(npz_fpath):
    X = np.load(npz_fpath)
    O = X[OBS].astype(np.float64)
    T= {}
    for k, v in X.items():
        if k != OBS:
            T[k] = v.astype(np.float64)
    return DictTensorDataset(O, T)


def summary_str(targets_dict):
    """
    Convenience routine for converting target dicts into string form
    :param targets_dict:
    :return:
    """
    if targets_dict is None:
        return ""
    ret = ""
    for var_name, var_value in targets_dict.items():
        if not(var_name == OBS):
            if not(isinstance(var_value, float)):
                var_value = ensure_numpy(var_value).flatten()[0]
            ret += "{}:\t{:.5f}\n".format(var_name, var_value)
    return ret


def unpack(datum, device="cpu"):
    """ Unpacks the given datum into Observation and Targets, primarily
    used to herd data from DictTensorDataset"""
    if isinstance(datum, torch.Tensor):
        O = datum
        T = None
    elif isinstance(datum, list):
        O = datum[0]
        T = datum[1:]
    elif isinstance(datum, dict):
        O = datum[OBS]
        T = {k: v.to(device) for k, v in datum.items()
              if k != OBS}
    else:
        raise Exception("Unsupported datum type={}".format(type(datum)))
    return O, T

#
# Convenience routines for formatting observations, etc...
#
def bformat_obs(datum, device="cpu"):
    """
    Batch formats the datum's observation, setting it ready for use by
    a model.
    """
    O = np.expand_dims(datum[OBS], 0)
    O = ensure_torch(device, O)
    return O


def format_batch(datum, device):
    """ Convenience routine for converting a batched input datum into
    batch suitable for training."""
    if isinstance(datum, torch.Tensor):
        O = datum
        Ts = {}
    elif isinstance(datum, list):
        O = datum[0]
        Ts = [D.to(device) for D in datum[1:]]  # Additional targets beyond reconstruction targets
    elif isinstance(datum, dict):
        O = datum[OBS]
        Ts = {k: v.to(device) for k, v in datum.items()
              if k != OBS}
    else:
        raise Exception("Unsupported datum type={}".format(type(datum)))
    return O, Ts


def forward(model, datum):
    """
    Given a datum instance, formats it as a batch and
    returns the result using the given model
    """
    O = bformat_obs(datum, model.device)
    return model(O)
