#
# Contains utility routines for working with PySC2 Episodes
#

from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset
from sc2recorder.utils import ensure_torch, ensure_numpy

from data.datasets import get_obs_tensors

def segment(observation_seq):
    """ Given a sequence of observations, splits into individual trajectories.  This 
    is done because each episode file in SC2 has multiple episodes in place.
    This is set for the basic battle scenario, where units decrease until one side
    is exhausted.  
    """
    episodes = []
    prev_alliance_hist = None
    scene_accum = []
    for obs in observation_seq:
        alliance_hist = Counter([x.alliance for x in obs['data']['raw_units']])
        if prev_alliance_hist is not None:
            prev_alliance_hist.subtract(alliance_hist)
            episode_terminated = False
            for alliance, delta in prev_alliance_hist.items():
                if delta < 0:
                    episode_terminated = True
            if episode_terminated:
                episodes.append(scene_accum)
                scene_accum = []
            scene_accum.append(obs)
        prev_alliance_hist = alliance_hist
    return episodes

def make_trajectory_ds(orig_episodes, encoder=None, device="cpu", use_binary_features=True):
    if encoder is not None:
        encoder.eval()
    dataset = TrajectoryDataset()
    for obs_seq in tqdm(orig_episodes):
        for obs in obs_seq:
            X, Y, C = get_obs_tensors(obs)
            X = ensure_torch(device, X).unsqueeze(0)
            C = ensure_torch(device, C).unsqueeze(0)
            if encoder is not None:
                obs['embedding'] = ensure_numpy(encoder.embed(X, C))
        dataset.add(obs_seq)
    return dataset

class TrajectoryDataset(Dataset):
    """ Contains sequences of observations corresponding to individual
    battle episodes.  This in turn is used to initialize different
    sequence characterization Datasets.
    """
    def __init__(self):
        super(TrajectoryDataset, self).__init__()
        self.add_obs = True
        self.episodes = []
    
    def __len__(self):
        return len(self.episodes)
       
    def __getitem__(self, idx):
        ep_obs = self.episodes[idx]
        ret = []
        for obs in ep_obs:
            X, Y, C = get_obs_tensors(obs)
            ret.append((X, Y, C, obs))
        return ret
    
    def get_raw(self, idx):
        return self.episodes[idx]
    
    def add(self, observation_seq):
        self.episodes.extend(segment(observation_seq))
        
 
    
    