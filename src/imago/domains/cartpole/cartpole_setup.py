import pandas as pd
import numpy as np
from gym.spaces.box import Box
from PIL import Image, ImageDraw, ImageFont
import torch

from imago import *
from imago.domains import ImagoDomain
from imago.io_utils import load_object
from pathlib import Path
from imago.models.model import PVaeModel, ModelType
from imago.model_utils import load_checkpoint
from imago.utils import ensure_numpy, tile_images_horiz
from imago.domains import DictTensorDataset
from imago.spaces import SPACE_TYPE, MCBoxSpec
from imago.spaces.channels import ChannelSpec
from tqdm import tqdm

OBSERVATION = 'observation'
ACTION_DIST = 'action_dist'
ACTION_LOGITS = 'action_logits'
VALUE_FUNCTION = 'value_function'

CONFIDENCE = "Confidence"
GOAL_COND = "Goal Conduciveness"
RISKINESS = "Riskiness"
INCONGRUITY = "Incongruity"

CHANNEL_SPECS = [
    ChannelSpec("position_cart", SPACE_TYPE.NUMERIC, low=-3, high=3, apply_whitening=True),
    ChannelSpec("velocity_cart", SPACE_TYPE.NUMERIC, low=-3, high=3, apply_whitening=True),
    ChannelSpec("angle_pole", SPACE_TYPE.NUMERIC, low=-3, high=3, apply_whitening=True),
    ChannelSpec("rotation_rate_pole", SPACE_TYPE.NUMERIC, low=-3, high=3, apply_whitening=True)
]

# Because this is a MLP domain, set the spatial dimensions to be (1, 1)
obs_space = Box(-3, 3, shape=(1, 1, 4))
mcbox_spec = MCBoxSpec(CHANNEL_SPECS, obs_space, channel_axis=-1)

LATENT2TARGET_SPECS = [
    ChannelSpec(VALUE_FUNCTION, SPACE_TYPE.NUMERIC, shape=(1,), mean=104.812, std=7.611, apply_recenter=True),
    ChannelSpec(CONFIDENCE, SPACE_TYPE.NUMERIC, shape=(1,)),
    ChannelSpec(GOAL_COND, SPACE_TYPE.NUMERIC, shape=(1,)),
    ChannelSpec(RISKINESS, SPACE_TYPE.NUMERIC, shape=(1,)),
    ChannelSpec(INCONGRUITY, SPACE_TYPE.NUMERIC, shape=(1,))
]

CAML_ROOT = Path(PVAE_ROOT, "..", "..")
DATA_ROOT = Path(CAML_ROOT, "datasets", "cartpole")
DEFAULT_DATA_FPATH = Path(DATA_ROOT, "interaction_data.pkl.gz")
DEFAULT_CSV_FPATH = Path(DATA_ROOT, "interestingness.csv.gz")

DEFAULT_MODEL_FPATH = Path(CAML_ROOT, "imago", "models", "cartpole_spatial_v2", "model.pt")

def _pad_fake_spatial_dim(X):
    """
    Pad with fake spatial dims so convolutional architecture works
    :param X:
    :return:
    """
    orig_shape = X.shape
    updated_shape = (orig_shape[0], 1, 1) + orig_shape[1:]
    #updated_shape = orig_shape + (1, 1)
    return X.reshape(updated_shape)

def load_data(data_fpath=DEFAULT_DATA_FPATH, intr_csv_fpath=DEFAULT_CSV_FPATH,
              tt_splitat=0.8, add_spatial_dims=True):
    rollouts = load_object(data_fpath)
    tt_rollout_index = int(len(rollouts) * tt_splitat)
    tt_eidx = 0
    assert tt_rollout_index > 0
    vars_df = pd.read_csv(intr_csv_fpath)  # Interestingness variables dataframe
    Os = []
    targets = {
        VALUE_FUNCTION: [],
        CONFIDENCE: [],
        GOAL_COND: [],
        RISKINESS: [],
        INCONGRUITY: []
    }

    for rollout_num, (rid, rollout) in tqdm(enumerate(rollouts.items())):
        """
        Unbox the Rollout and Interaction Data
        """
        rid = int(rid)  # CSV lookup uses longs as rollout IDs
        intr_df = vars_df[vars_df.Rollout == rid]  # TODO: Assert these exist
        O = rollout.data.observation  # (Frames, ... )
        R = rollout.data.reward
        V = rollout.data.value
        assert len(intr_df) == len(O) == len(R) == len(V)

        # Unroll the episode into individual instances
        Os.append(O)
        targets[VALUE_FUNCTION].append(V)
        for colname in [CONFIDENCE, GOAL_COND, RISKINESS, INCONGRUITY]:
            arr = intr_df[colname].to_numpy().astype('float32').reshape((-1, 1))  # Padd with a column value so vstack works
            targets[colname].append(arr)
        # Identify episode split point
        if rollout_num < tt_rollout_index:
            tt_eidx += len(O)

    assert tt_eidx > 0
    Os_train, Os_test = np.vstack(Os[0:tt_rollout_index]), np.vstack(Os[tt_rollout_index:])
    if add_spatial_dims:
        Os_train = _pad_fake_spatial_dim(Os_train)
        Os_test = _pad_fake_spatial_dim(Os_test)

    # Add false spatial dimensions to get this to work with our current Conv setup
    targets_train, targets_test = {}, {}
    for k, v in targets.items():
        targets_train[k] = np.vstack(v[0:tt_rollout_index])
        targets_test[k] = np.vstack(v[tt_rollout_index:])

    train_dataset = DictTensorDataset(Os_train, targets_train)
    test_dataset = DictTensorDataset(Os_test, targets_test)
    return train_dataset, test_dataset


def load_model(model_fpath=DEFAULT_MODEL_FPATH, device="cuda:0"):
    model = PVaeModel(mcbox_spec, latent2target_specs=LATENT2TARGET_SPECS,
                      device=device).to(device)
    if Path(model_fpath).exists():
        load_checkpoint(model_fpath, model, None, device)
    return model

X_THRESHOLD = 2.4

def load_domain(device="cpu"):
    model = load_model(device=device)
    return ImagoDomain(model, compute_Odiff, render_cartpole)


def render_cartpole(O, V=None):
    imgs = []
    for idx in range(len(O)):
        o = O[idx]
        if V is not None:
            v = V[idx]
        else:
            v = None
        imgs.append(_render_single(o, v))
    return tile_images_horiz(imgs)

def _render_single(o, v=None):
    o = ensure_numpy(o).flatten()
    if len(o.shape) == 5:
        o = o[0] # Remove batch dimension, get just the first entry
    assert len(o) == 4

    # Rest appropriated from cartpole v0 gym env
    #screen_width, screen_height = 600, 400
    screen_width, screen_height = 400, 300
    world_width = X_THRESHOLD * 2
    scale = screen_width / world_width
    cart_width, cart_height = 50, 30
    pole_length = 2 * scale
    cart_pos = o[0]
    cart_vel = o[1]
    pole_angle = o[2]
    pole_rotation_rate = o[3]
    cart_x = cart_pos * scale + screen_width / 2.0
    cart_y = screen_height - 100
    cart_left = cart_x - cart_width / 2
    cart_right = cart_x + cart_width / 2
    cart_top = cart_y - cart_height / 2
    cart_bottom = cart_y + cart_height / 2
    cart_color = (200, 200, 100)

    pole_tip_x = cart_x + np.sin(pole_angle) * pole_length
    pole_tip_y = cart_top - np.cos(pole_angle) * pole_length
    pole_color = (100, 200, 200)
    img = Image.new("RGB", (screen_width, screen_height), color=(255, 255, 255))
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    drawer = ImageDraw.Draw(img)
    drawer.rectangle(((cart_left, cart_top), (cart_right, cart_bottom)), fill=cart_color)
    drawer.line(((cart_x, cart_top), (pole_tip_x, pole_tip_y)), width=10, fill=pole_color)
    drawer.text((10, 10), "Pos={:.3f}, vel={:.3f}".format(cart_pos, cart_vel),
                font=font, fill=(0,0,0))
    drawer.text((10, 30), "Angle={:.3f}, Rot-rate={:.3f}".format(pole_angle,
                                                                 pole_rotation_rate),
                font=font, fill=(0,0,0))
    if v is not None:
        drawer.text((10, 50), "Value={:.5f}".format(v.flatten()[0]))
    return img


def compute_Odiff(O1, O2):
    O1, O2 = ensure_numpy(O1), ensure_numpy(O2)
    return np.sum(np.abs(O1 - O2))