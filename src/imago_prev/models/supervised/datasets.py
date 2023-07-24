"""
Datasets used for supervised training
"""
import random
from absl import logging
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sc2scenarios.scenarios.simple1 import vec2var, perturb1


class FirstObsDataset(Dataset):
    """
    Used for predicting the first observations of trajectories.  This is so we can get snapshots
    corresponding to the starting conditions.
    """
    def __init__(self, trajectory_dataset, n=1):
        self.trajectory_dataset = trajectory_dataset
        self.metadata = trajectory_dataset.metadata
        self.n = n

    def __getitem__(self, item):
        """ Gets a frame from randomly within the first N observations of the given trajectory """
        traj_X, traj_Y, traj_C = self.trajectory_dataset[item]
        idx = random.randint(0, min(len(traj_X), self.n))
        return traj_X[idx], traj_Y[idx], traj_C[idx]

    def __len__(self):
        return len(self.trajectory_dataset)


def vec2key(vec):
    return np.array2string(vec)


class PerturbDataset(Dataset):
    """
    Given a backing FirstObsDataset, generates paired instances based on the conditioning variables.  The conditioning for the second supervised interpolation is the delta in conditioning variables.
    """
    def __init__(self, first_obs_dataset):
        self.first_obs_dataset = first_obs_dataset
        self.C_lookup = {}
        for idx, (X, Y, C) in tqdm(enumerate(self.first_obs_dataset)):
            C_key = vec2key(C)
            if C_key not in self.C_lookup:
                self.C_lookup[C_key] = []
            self.C_lookup[C_key].append(idx)
        logging.info("CondVarDataset, total # C keys={}".format(len(self.C_lookup)))

    def __getitem__(self, item):
        X1, Y1, C1 = self.first_obs_dataset[item]
        c_key = None
        while (c_key is None) or (c_key not in self.C_lookup):
            C2 = perturb1(*vec2var(C1))
            c_key = vec2key(C2)
        inst2_idxes = self.C_lookup[c_key]
        inst2_idx = random.choice(inst2_idxes)
        X2, Y2, C2 = self.first_obs_dataset[inst2_idx]
        C_d = C2 - C1
        return X1, Y1, C1, X2, Y2, C_d

    def __len__(self):
        return len(self.first_obs_dataset)
    

    
    

