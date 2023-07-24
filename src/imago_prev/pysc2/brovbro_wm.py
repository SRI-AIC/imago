""" Formats data for WorldModels Pytorch platform (https://github.com/ctallec/world-models)
"""
import numpy as np
# from logger import Logger
from tqdm import tqdm

import os
from imago_prev.models.model_util import safe_mkdirs
from data.datasets import process_episode_dir
from sc2recorder.utils import set_random_seed

#from imago.models.csvae import *
from imago_prev.pysc2.episode_utils import make_trajectory_ds

SMALL_DATA=True  # More for which model to load
USE_BINARY_FEATURES=False

N_LATENT   = 10
N_HIDDEN = 10
device="cuda:0"

ROOT_DIR="."

train_dir = "{}/data/data_combat1".format(ROOT_DIR)
output_data_dir = "{}/data/wm".format(ROOT_DIR)
safe_mkdirs(output_data_dir)

binary_label="binary"
if not(USE_BINARY_FEATURES):
    binary_label="rvf"
size_label="small"
if not(SMALL_DATA):
    size_label="large"

set_random_seed()

episodes, metadata = process_episode_dir(train_dir, USE_BINARY_FEATURES=False)
traj_dataset = make_trajectory_ds(episodes, encoder=None, device="cpu", use_binary_features=USE_BINARY_FEATURES)

for episode_idx in tqdm(range(len(traj_dataset))):
    s_rollout = [] # state rollout
    r_rollout = [] # reward
    d_rollout = [] # Done indicator
    a_rollout = [] # Action, usually sampled, but here we just set to 0
    for X, Y, C, obs in traj_dataset[episode_idx]:
        obs_img = obs['data']['rgb_screen']
        obs_img = obs_img.resize((64, 64))  # Force to target dimensions
        s_rollout.append(np.array(obs_img))  # B/c of torch transforms, may have to convert into Pytorch normal form
        r_rollout.append(0)
        d_rollout.append(0)
        a_rollout.append(np.zeros(3))  # Done to match ASIZE specified in world model code
    if len(d_rollout) >= 5:
        # Avoid small trajectories
        d_rollout[-1] = 1
        np.savez(os.path.join(output_data_dir, 'rollout_{}'.format(episode_idx)),
                      observations=np.array(s_rollout),
                      rewards=np.array(r_rollout),
                      actions=np.array(a_rollout),
                      terminals=np.array(d_rollout))
    