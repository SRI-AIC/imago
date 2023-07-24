from absl import logging

from imago.spaces.channels import ChannelSpec

logging.set_verbosity(logging.INFO)

from imago import PVAE_ROOT
from imago.models.model import PVaeModel
from imago.spaces.mc_box import *
from gym.spaces.box import Box
import os
from glob import glob
from tqdm import tqdm
from interestingness_xdrl.util.io import load_object
from skimage import measure as ski_measure
from multiprocessing.pool import Pool

from imago.utils import *
from imago.losses import model_loss

""" Test the updated architecture on the SC2 frames.
"""

DEBUG = True

print(os.getcwd())

CAML_ROOT = os.path.join(PVAE_ROOT, "..", "..")
IDP_ROOT = os.path.join(CAML_ROOT, "output", "assault", "interaction_data")

DEVICE='cpu'

def downsample_frame(frame):
    F2 = ski_measure.block_reduce(frame, (1, 2, 3), func=np.max)
    F2a = F2[:, 4:68, :]  # Note: Strips away top 4 and bottom 4 rows
    return F2a


obs_space = Box(0, 2048, shape=(64, 64, 3), dtype=np.uint8)

# Specify SC2 components
CHANNEL_SPECS = [
    ChannelSpec("player_relative", SPACE_TYPE.CATEGORICAL, low=0, high=4),
    ChannelSpec("unit_type", SPACE_TYPE.CATEGORICAL, low=0, high=2048),
    ChannelSpec("unit_hit_points_ratio", SPACE_TYPE.SPARSE_NUMERIC, low=0, high=255)
]

mcbox_spec = MCBoxSpec(CHANNEL_SPECS, obs_space, channel_axis=1)

model = PVaeModel(mcbox_spec, device=DEVICE)

idp_files = glob(os.path.join(IDP_ROOT, "**", "**", "interaction_data.pkl.gz"))

if DEBUG:
    idp_files = idp_files[0:10]

print("Root={}, Total IDP files={}".format(os.path.abspath(IDP_ROOT), len(idp_files)))

class IDPResult:
    def __init__(self, idp, idp_fpath):
        self.idp, self.idp_fpath = idp, idp_fpath

def load_idp(idp_fpath):
    return IDPResult(load_object(idp_fpath), idp_fpath)

with Pool(10) as p:
    # Attempt to load each episode.  Remove failures (value None).
    trajectories = list(tqdm(p.imap_unordered(load_idp, idp_files), total=len(idp_files)))

print("Total # trajectories = {}".format(len(trajectories)))

# Flatten all IDPs and assemble scene recon dataset
# TODO: Associate these with their feature file
all_idps = []
for trajectory in tqdm(trajectories):
    frame_idps = trajectory.idp
    for frame_idp in frame_idps:
        frame_idp.fpath = trajectory.idp_fpath
        frame_idp.observation = downsample_frame(frame_idp.observation)
        all_idps.append(frame_idp)
print(len(all_idps))

X = all_idps[0].observation
O = ensure_torch_rec(DEVICE, X).unsqueeze(0)
loss, named_losses = model_loss(model, O)
