import numpy as np
import hickle as hkl
from gym.spaces.box import Box
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from imago import *
from imago.domains import DictTensorDataset
from imago.models.model import PVaeModel
from imago.spaces import SPACE_TYPE, MCBoxSpec
from imago.spaces.channels import ChannelSpec
from imago.utils import sigmoid
from imago.viz import CATEGORICAL_COLORS, NUMERIC_RANGE_COLORS
from imago.model_utils import load_checkpoint

OBSERVATION = 'observation'
ACTION_DIST = 'action_dist'
ACTION_LOGITS = 'action_logits'
VALUE_FUNCTION = 'value_function'

IMAGO_ROOT = os.path.join(PVAE_ROOT, "../..")
CAML_ROOT = os.path.join(PVAE_ROOT, "../..", "..")
DEFAULT_DATA_FILE = os.path.join(CAML_ROOT, "datasets", "cameleon", "imago_rollouts-2578_rs42_w14.hkl")
DEFAULT_OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/pvae/demos/canniballs_2578_rs42_w14")
DEFAULT_MODEL_FPATH = os.path.join(IMAGO_ROOT, "models", "canniballs_2578_rs42_w14", "model.pt")

obs_space = Box(0, 2048, shape=(12, 12, 4), dtype=np.uint8)

CHANNEL_SPECS = [
    ChannelSpec("agent_pos", SPACE_TYPE.CATEGORICAL, low=0, high=1),
    ChannelSpec("opponents", SPACE_TYPE.CATEGORICAL, low=0, high=1),    
    ChannelSpec("food", SPACE_TYPE.CATEGORICAL, low=0, high=1),
    ChannelSpec("obstacles", SPACE_TYPE.CATEGORICAL, low=0, high=1)    
    ]
    
LATENT2TARGET_SPECS = [
    ChannelSpec("action_dist", SPACE_TYPE.DIST_JSD, shape=(4,)),
    ChannelSpec("action_logits", SPACE_TYPE.DIST_JSD, shape=(4,), low=0, high=1),
    ChannelSpec("value_function", SPACE_TYPE.NUMERIC, shape=(1,), low=0, high=1, apply_whitening=True)
]
    
mcbox_spec = MCBoxSpec(CHANNEL_SPECS, obs_space, channel_axis=-1)

def load_canniballs_model(model_fpath=DEFAULT_MODEL_FPATH, device="cuda:0"):
    model = PVaeModel(mcbox_spec, latent2target_specs=LATENT2TARGET_SPECS,
                  device=device)
    load_checkpoint(model_fpath, model, None)
    return model

def load_canniballs_data(hkl_fpath=DEFAULT_DATA_FILE, batch_size=64):
    D = hkl.load(hkl_fpath)
    O = D[OBSERVATION]
    A = D[ACTION_DIST]
    L = D[ACTION_LOGITS]
    L = sigmoid(L)
    V = D[VALUE_FUNCTION]
    V = V.reshape((V.shape[0], 1))

    val_idx = -1000
    train_tensors = []
    test_tensors = []

    # Standardize
    v_scaler = StandardScaler()
    v_scaler.fit(V[0:val_idx])
    V = v_scaler.transform(V)

    for X in [O, A, L, V]:
        train_tensors.append(torch.Tensor(X[0:val_idx]))
        test_tensors.append(torch.Tensor(X[val_idx:]))

    T_train = {}
    for name, M in [(ACTION_DIST, A[0:val_idx]),
                    (ACTION_LOGITS, L[0:val_idx]),
                    (VALUE_FUNCTION, V[0:val_idx])]:
        T_train[name] = M
    train_ds = DictTensorDataset(O[0:val_idx], T_train)

    T_val = {}
    for name, M in [(ACTION_DIST, A[val_idx:]),
                    (ACTION_LOGITS, L[val_idx:]),
                    (VALUE_FUNCTION, V[val_idx:])]:
        T_val[name] = M
    test_ds = DictTensorDataset(O[val_idx:], T_val)
    
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader




#
# Canniballs specific visualization code
#
FOOD_COLORS = [(0,0,0), (100,255,100)]
AGENT_COLORS = [(0,0,0), (100,100,255)]
ENEMY_COLORS = [(0,0,0), (255, 100, 100)]

CANNIBALLS_PALETTES = [AGENT_COLORS, ENEMY_COLORS, FOOD_COLORS]

def plot_2d_map_stacked(arr, channel_spec,
                        drawer,  # Existing image to draw on
                name=None, cell_width=2, bgcolor=(255,255,255),
                palette=None,
                add_legend=False,
                legend_text_size=10, legend_offset=2):
    """ Given a HxW array with integers, plots and returns an image, treating each
    of the integers as a categorical label.  If a lookup is given, will plot this
    below as well.  If categorical, uses only the 
    NOTE: PIL is (x,y), while arrays are (y, x).  We display in (x,y) form."""
    if len(arr.shape) != 2:
        raise ValueError("Input array should have shape length 2 (single image).  Given input shape={}".format(arr.shape))
    arr = ensure_numpy(arr)
    h, w = arr.shape
    distinct_labels = np.unique(arr)
    if add_legend:
        if not(channel_spec.is_numeric):
            legend_height = (1 + len(distinct_labels)) * (legend_text_size + legend_offset)
        else:
            legend_height = (1 + 2) * (legend_text_size + legend_offset)
    else:
        legend_height = 0
        legend_offset = 0
    try:
        label_font = ImageFont.truetype("arial.ttf", legend_text_size)
    except OSError:
        label_font = ImageFont.load_default()
    distinct_labels = set()
    # If no palette, assign palette based on categorical or numeric
    if palette is None:
        if channel_spec.is_numeric or channel_spec.is_distributional:
            palette = NUMERIC_RANGE_COLORS
        elif channel_spec.space_type == SPACE_TYPE.CATEGORICAL:
            palette = CATEGORICAL_COLORS
        else:
            raise Exception("Unsupported channel space type={}".format(channel_spec.space_type))

    for row in range(h):
        for col in range(w):
            x, y = col, row
            x *= cell_width
            y *= cell_width
            label = int(arr[row, col])
            distinct_labels.add(label)
            if label > 0:
                if channel_spec.is_numeric:
                    label_offset = max(0, int(round((label / channel_spec.size) * len(palette)) - 1))
                    color = palette[label_offset % len(palette)]
                elif channel_spec.is_distributional:
                    label_offset = max(0, int(round(label * len(palette)) - 1))  # Rescale to top of numeric palette
                    color = palette[label_offset % len(palette)]
                else:
                    color = palette[label % len(palette)]
                drawer.rectangle(((x, y), (x + cell_width, y + cell_width)), fill=color)
    x = 5
    y = h * cell_width + legend_offset
    if add_legend:
        if name is None:
            name = channel_spec.name
        drawer.text((x, y), "{}".format(name), font=label_font, fill=(0,0,0))
        if not(channel_spec.is_numeric):
            # Plot the categories
            for dlabel in distinct_labels:
                y += legend_text_size + legend_offset
                drawer.rectangle(((x, y), (x + 50, y + legend_text_size)), fill=palette[dlabel % len(palette)])
                drawer.text((x + 52, y), "{}".format(dlabel), font=label_font, fill=(0,0,0))
        elif channel_spec.is_distributional:
            # Plot just the low and high
            for dlabel in [0, channel_spec.size]:
                dlabel_offset = max(0, round((dlabel / channel_spec.size) * len(palette)) - 1)
                y += legend_text_size + legend_offset
                drawer.rectangle(((x, y), (x + 50, y + legend_text_size)), fill=palette[dlabel_offset % len(palette)])
                drawer.text((x + 52, y), "{}".format(dlabel), font=label_font, fill=(0, 0, 0))
        else:
            # Plot just the low and high
            for dlabel in [0, channel_spec.size]:
                dlabel_offset = max(0, round((dlabel / channel_spec.size) * len(palette)) - 1)
                y += legend_text_size + legend_offset
                drawer.rectangle(((x, y), (x + 50, y + legend_text_size)), fill=palette[dlabel_offset % len(palette)])
                drawer.text((x + 52, y), "{}".format(dlabel), font=label_font, fill=(0,0,0))

def render_stacked(O, space_specifier, cell_width=2, bgcolor=(0,0,0)):
    """ Given the batched observation and the space specification,
    constructs the most appropriate image summaries for the
    images in this observation batch."""
    palettes = [AGENT_COLORS, ENEMY_COLORS, FOOD_COLORS]
    if isinstance(space_specifier, MCBoxSpec):
        batch_imgs = []
        for batch_idx in range(len(O)):
            for channel_spec in space_specifier.channel_specs:
                o1 = space_specifier.get_channel_slice(O, cidx=channel_spec.channel_idx,
                                                       batch_idx=batch_idx)
                break
            h, w = o1.shape
            legend_height = 0
            img = Image.new("RGB", (w * cell_width, h * cell_width + legend_height), bgcolor)
            drawer = ImageDraw.Draw(img)
            drawer.rectangle(((0, 0), (w, h)), fill=bgcolor)
            inst_imgs = []
            for channel_spec, palette in zip(space_specifier.channel_specs, palettes):
                o1 = space_specifier.get_channel_slice(O, cidx=channel_spec.channel_idx,
                                                       batch_idx=batch_idx)
                plot_2d_map_stacked(o1, channel_spec, drawer, cell_width=cell_width, palette=palette)
            batch_imgs.append(img)
        return tile_images_grid(batch_imgs, num_per_row=1)
    else:
        raise Exception("Unsupported space type={}".format(type(space_specifier)))
