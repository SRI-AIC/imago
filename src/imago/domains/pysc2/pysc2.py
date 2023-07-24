import os
from tqdm import tqdm
from glob import glob
from skimage import measure as ski_measure
from multiprocessing.pool import Pool
from imago.io_utils import load_object
from imago_prev.data.datasets import IDPDataset

#from imago.data.datasets import VALUE, ACTION_PROBS

from imago import PVAE_ROOT, IMAGO_ROOT, CAML_ROOT
from imago.models.model import PVaeModel
from imago.spaces.mc_box import *
from gym.spaces.box import Box

# from imago import *


from imago.analysis import interests
from imago.spaces.mc_box import MCBoxSpec
from imago.spaces.channels import SPACE_TYPE, ChannelSpec
from imago.model_utils import load_checkpoint


OBS = "observation"
VALUE = "value_function"
ACTION_PROBS = "action_probs_{}"

"""
CAML PySC2 specific routines for formatting and loading a dataset
"""

# CAML_ROOT = os.path.join(PVAE_ROOT, "../..", "..")
CAML_Y1_IDP_ROOT = os.path.join(CAML_ROOT, "datasets", "CamlYear1Eval", "assault", "interaction_data")

obs_space = Box(0, 2048, shape=(3, 64, 64), dtype=np.uint8)
OBS_CHANNEL_SPECS = [
    ChannelSpec("player_relative", SPACE_TYPE.CATEGORICAL, low=0, high=4),
    ChannelSpec("unit_type", SPACE_TYPE.CATEGORICAL, low=0, high=2048),
    ChannelSpec("unit_hit_points_ratio", SPACE_TYPE.SPARSE_NUMERIC, low=0, high=255)
]
mcbox_spec = MCBoxSpec(OBS_CHANNEL_SPECS, obs_space, channel_axis=0)

LATENT2_TARGET_SPECS = [
    ChannelSpec(VALUE, SPACE_TYPE.NUMERIC,
                shape=(1,),
                low=0, high=1)
]

# Distributions governing behaviors
Y1_EVAL_Bs = [ 4, 4, 4, 4, 24, 24, 24, 24, 6, 6, 6, 6]

# Offset in the tensor vector for value, then action indices
value_idx=0
action_idxes=[i for i in range(1, 13)]

interests_profile = interests.InterestsProfile(action_idxes, value_idx)

for idx, b_dim in enumerate(Y1_EVAL_Bs):
    LATENT2_TARGET_SPECS.append(ChannelSpec(ACTION_PROBS.format(idx),
                                            SPACE_TYPE.DIST_JSD,
                                            shape=(b_dim, ),
                                            low=0, high=1))


def downsample_frame(frame):
    F2 = ski_measure.block_reduce(frame, (1, 2, 3), func=np.max)
    F2a = F2[:, 4:68, :]  # Note: Strips away top 4 and bottom 4 rows
    return F2a


class IDPResult:
    def __init__(self, idp, idp_fpath):
        self.idp, self.idp_fpath = idp, idp_fpath

def load_idp(idp_fpath):
    return IDPResult(load_object(idp_fpath), idp_fpath)


def load_idps(idp_files):
    """
    :param:idp_files:
    :return:
    """
    with Pool(10) as p:
        # Attempt to load each episode.  Remove failures (value None).
        trajectories = list(tqdm(p.imap(load_idp, idp_files), total=len(idp_files)))        
    all_idps = []
    for trajectory in tqdm(trajectories):
        frame_idps = trajectory.idp
        for frame_idp in frame_idps:
            frame_idp.fpath = trajectory.idp_fpath
            frame_idp.observation = downsample_frame(frame_idp.observation)
            all_idps.append(frame_idp)
    return all_idps


def load_traj_idps(idp_files):
    """
    Similar to load_idps, but keeps trajectory structure
    :param:idp_files:
    :return:
    """
    with Pool(10) as p:
        # Attempt to load each episode.  Remove failures (value None).
        trajectories = list(tqdm(p.imap(load_idp, idp_files), total=len(idp_files)))        
#        trajectories = list(tqdm(p.imap_unordered(load_idp, idp_files), total=len(idp_files)))
    all_trajs = []
    for trajectory in tqdm(trajectories):
        all_idps = []        
        frame_idps = trajectory.idp
        for frame_idp in frame_idps:
            frame_idp.fpath = trajectory.idp_fpath
            frame_idp.observation = downsample_frame(frame_idp.observation)
            all_idps.append(frame_idp)
        all_trajs.append(all_idps)
    return all_trajs



def load_y1_data(debug=False):
    """
    Loads in the Caml Y1 evaluation data
    """
    CAML_ROOT = os.path.join(IMAGO_ROOT, "../..")
    IDP_ROOT = CAML_Y1_IDP_ROOT
    idp_files = glob(os.path.join(IDP_ROOT, "**", "**", "interaction_data.pkl.gz"))
    if debug:
        idp_files = idp_files[0:10]
    print("Root={}, Total IDP files={}".format(os.path.abspath(IDP_ROOT), len(idp_files)))
    all_idps = load_idps(idp_files)
    train_dataset = IDPDataset(all_idps[0:-100])
    v_mean, v_std = train_dataset.compute_V_params()
    test_dataset = IDPDataset(all_idps[-100:])
    test_dataset.set_V_params(v_mean, v_std)
    return train_dataset, test_dataset, v_mean, v_std


def load_y1_as_trajs(debug=False, split_at=0.8):
    """ Similar to load_y1_data, but separates by trajectories instead
    """ 
    CAML_ROOT = os.path.join(IMAGO_ROOT, "../..")
    IDP_ROOT = CAML_Y1_IDP_ROOT
    idp_files = glob(os.path.join(IDP_ROOT, "**", "**", "interaction_data.pkl.gz"))
    if debug:
        idp_files = idp_files[0:10]
    print("Root={}, Total IDP files={}".format(os.path.abspath(IDP_ROOT), len(idp_files)))
    all_trajs = load_traj_idps(idp_files)
    split_idx = int(len(all_trajs) * split_at)
    assert split_idx > 0 and split_idx < len(all_trajs)
    train_trajs = all_trajs[0:split_idx]
    test_trajs = all_trajs[split_idx:]
    
    train_dataset = IDPDataset(train_trajs)
    v_mean, v_std = train_dataset.compute_V_params()
    test_dataset = IDPDataset(test_trajs)
    test_dataset.set_V_params(v_mean, v_std)
    return train_dataset, test_dataset, train_trajs, test_trajs


def setup_y1_model(device="cpu"):
    model = PVaeModel(mcbox_spec, latent2target_specs=LATENT2_TARGET_SPECS,
                      device=device).to(device)
    return model


def load_model(model_fpath, device="cpu"):
    model = setup_y1_model(device=device)
    load_checkpoint(model_fpath, model, None)
    return model
