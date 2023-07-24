"""
Alternate visualization package, similar to viz.py, but allows
multiple images to be drawn out, as well as customized palettes.
"""

from PIL import Image, ImageDraw, ImageFont
from imago.viz import CATEGORICAL_COLORS, NUMERIC_RANGE_COLORS
from imago.spaces.mc_box import *

from imago.utils import *

DEVICE="cuda:0"

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

def render_stacked(O, space_specifier, palettes, cell_width=2, bgcolor=(0,0,0)):
    """ Given the batched observation and the space specification,
    constructs the most appropriate image summaries for the
    images in this observation batch."""
    assert not(isinstance(O, dict))  # We need to be working with the tensor version of the observation
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
