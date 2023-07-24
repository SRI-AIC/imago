import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from tqdm import tqdm

from imago.domains import OBS, summary_str
from imago.spaces import MCBoxSpec
from imago.spaces.channels import SPACE_TYPE
from imago.utils import tile_images_grid, tile_images_horiz, ensure_numpy

"""
Routines for displaying outputs for different types of
observations.
"""

crng = random.Random()
crng.seed(1337)
CATEGORICAL_COLORS = []
for n in range(1000):
    vec = np.array([crng.random() for x in range(3)])
    vec /= np.linalg.norm(vec)
    vec *= 255
    vec = np.rint(vec)
    CATEGORICAL_COLORS.append(tuple(int(x) for x in vec))


# Fixed positions for colors (0 = black, 1=blue, 4=red)
CATEGORICAL_COLORS[0] = (0,0,0)
CATEGORICAL_COLORS[1] = (10,10,255) # Player Friendly
CATEGORICAL_COLORS[4] = (255,10,10) # Opponent

# Range color wheel for numeric
NUMERIC_RANGE_COLORS=[(0,0,0)]
for (r,g,b) in plt.get_cmap('viridis').colors:
    NUMERIC_RANGE_COLORS.append((int(r*255), int(g*255), int(b*255)))


def render(O, space_specifier, cell_width=2, notes_arr=None):
    """ Given the batched observation and the space specification,
    constructs the most appropriate image summaries for the
    images in this observation batch.

    If a notes array is passed through notes_arr, then a separate
    pane writing out these notes is attached.  Currently the text
    will be fixed WxH, using the system default font."""
    assert not(isinstance(O, dict)) # This needs to be the observation stensor
    if notes_arr is not None:
        assert len(notes_arr) == len(O)  # Assert number of notes and images line up
    if isinstance(space_specifier, MCBoxSpec):
        batch_imgs = []
        for batch_idx in range(len(O)):
            inst_imgs = []
            for channel_spec in space_specifier.channel_specs:
                o1 = space_specifier.get_channel_slice(O, cidx=channel_spec.channel_idx,
                                                       batch_idx=batch_idx)
                img = plot_2d_map(o1, channel_spec, cell_width=cell_width)
                inst_imgs.append(img)
            if notes_arr is not None:
                note_txt = notes_arr[batch_idx]
                note_img = Image.new('RGB', size=(160, 120), color=(0, 0, 0))  # TODO:Dynamically a djust based on font
                txt_draw = ImageDraw.Draw(note_img)
                font = ImageFont.load_default()
                txt_draw.text((5, 5), note_txt, fill=(255, 255, 255), font=font)
                inst_imgs.append(note_img)
            batch_imgs.append(tile_images_grid(inst_imgs, num_per_row=4))
        return tile_images_grid(batch_imgs, num_per_row=1)
    else:
        raise Exception("Unsupported space type={}".format(type(space_specifier)))


def render2gif(Os, space_specifier, gif_fpath, cell_width=2,
               notes_arr=None,
               **kwd):
    """ Given a sequence of observations, constructs an animated
    sequence and saves to file."""
    if notes_arr is not None:
        assert len(Os) == len(notes_arr)
    imgs = []
    for idx in tqdm(range(len(Os))):
        note = None
        if notes_arr is not None:
            note = [notes_arr[idx]]
        imgs.append(render(np.expand_dims(Os[idx], 0),
               space_specifier=space_specifier,
               cell_width=cell_width,
               notes_arr = note,
               **kwd))
    imgs[0].save(gif_fpath, save_all=True, append_images=imgs[1:], duration=75, loop=0)


def render_ds2gif(dataset, start_idx, end_idx, space_specifier, gif_fpath, **kwd):
    Os = []
    notes_arr = []
    for idx in range(start_idx, end_idx+1):
        datum = dataset[idx]
        Os.append(datum[OBS])
        notes_arr.append("Step: {}\n{}".format(idx, summary_str(datum)))
    render2gif(Os, space_specifier, gif_fpath, notes_arr = notes_arr)


def plot_2d_map(arr, channel_spec,
                name=None, cell_width=2, bgcolor=(255,255,255),
                palette=None,
                add_legend=True,
                legend_text_size=10, legend_offset=2):
    """ Given a HxW array with integers, plots and returns an image, treating each
    of the integers as a categorical label.  If a lookup is given, will plot this
    below as well.
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
    img = Image.new("RGB", (w * cell_width, h * cell_width + legend_height), bgcolor)
    drawer = ImageDraw.Draw(img)
    distinct_labels = set()
    # If no palette, assign palette based on categorical or numeric
    if palette is None:
        if channel_spec.is_numeric or channel_spec.is_distributional:
            palette = NUMERIC_RANGE_COLORS
        elif channel_spec.space_type == SPACE_TYPE.CATEGORICAL or \
                channel_spec.space_type == SPACE_TYPE.BINARY:
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
    return img
