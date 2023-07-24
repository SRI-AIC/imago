import random
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import numpy as np
import math
import torch

from .frames import *

import pdb

crng = random.Random()
crng.seed(1337)
CATEGORICAL_COLORS = []
for n in range(1000):
    vec = np.array([crng.random() for x in range(3)])
    vec /= np.linalg.norm(vec) 
    vec *= 255
    vec = np.rint(vec)
    CATEGORICAL_COLORS.append(tuple(int(x) for x in vec))
    
# Magic positions
CATEGORICAL_COLORS[0] = (0,0,0)
CATEGORICAL_COLORS[1] = (10,10,255) # Player Friendly
CATEGORICAL_COLORS[4] = (255,10,10) # Opponent


# Range color wheel for numeric
NUMERIC_RANGE_COLORS=[(0,0,0)]
for (r,g,b) in plt.get_cmap('viridis').colors:
    NUMERIC_RANGE_COLORS.append((int(r*255), int(g*255), int(b*255)))


def plot_categorical_map(arr, name="grid", cell_width=2, bgcolor=(255,255,255), 
                         palette=CATEGORICAL_COLORS, 
                         add_legend=True,
                         legend_text_size=10, legend_offset=2,
                        is_numeric_key=True, numeric_max_val=255):
    """ Given a HxW array with integer labels, plots and returns an image.  If a lookup is given, will plot this
    below as well.
    NOTE: PIL is (x,y), while arrays are (y, x).  We display in (x,y) form."""
    if len(arr.shape) != 2:
        raise ValueError("Input array should have shape length 2.  Given input shape={}".format(arr.shape))
    h, w = arr.shape
    distinct_labels = np.unique(arr)
    if add_legend:
        if not(is_numeric_key):
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
    for row in range(h):
        for col in range(w):
            x, y = col, row
            x *= cell_width
            y *= cell_width
            label = arr[row, col]
            distinct_labels.add(label)
            if is_numeric_key:
                label_offset = max(0, int(round((label / numeric_max_val) * len(palette)) - 1))
                color = palette[label_offset % len(palette)]
            else:
                color = palette[label % len(palette)]
            drawer.rectangle(((x, y), (x + cell_width, y + cell_width)), fill=color)
    x = 5
    y = h * cell_width + legend_offset
    if add_legend:
        drawer.text((x, y), "{}".format(name), font=label_font, fill=(0,0,0))
        if not(is_numeric_key):
            # Plot the categories
            for dlabel in distinct_labels:
                y += legend_text_size + legend_offset
                drawer.rectangle(((x, y), (x + 50, y + legend_text_size)), fill=palette[dlabel % len(palette)])
                drawer.text((x + 52, y), "{}".format(dlabel), font=label_font, fill=(0,0,0))
        else:
            # Plot just the low and high
            for dlabel in [0, numeric_max_val]:
                dlabel_offset = max(0, round((dlabel / numeric_max_val) * len(palette)) - 1)
                y += legend_text_size + legend_offset
                drawer.rectangle(((x, y), (x + 50, y + legend_text_size)), fill=palette[dlabel_offset % len(palette)])
                drawer.text((x + 52, y), "{}".format(dlabel), font=label_font, fill=(0,0,0))
    return img


def _render(sc2obs, components, margin=20):
    img_dict = sc2obs['data']['feature_screen']
    name2imgs = {}
    summary_width = 0
    summary_height = 0
    for component in components:
        if hasattr(component, 'name'):
            field = component.name
        else:
            field = str(component)
        is_numeric = isinstance(component, NumericFrame) or \
                     isinstance(component, SparseNumericFrame) or \
                     str(component).endswith("ratio")
        if is_numeric:
            palette = NUMERIC_RANGE_COLORS
            is_numeric_key = True
            max_val = component.max_value
        else:
            palette = CATEGORICAL_COLORS
            is_numeric_key=False
            max_val = 255
        arr = np.array(img_dict[field]) # This needs to be done as NamedNumpyArrays have issues with Numpy functions
        img = plot_categorical_map(arr, name=field, palette=palette, is_numeric_key=is_numeric_key,
                                   numeric_max_val=max_val)
        name2imgs[field] = img
        width, height = img.size
        summary_width += width + margin
        summary_height = max(height, summary_height)
    summary_image = Image.new("RGB", (summary_width, summary_height),(255,255,255))
    curr_x, curr_y = margin, 0
    for name, img in name2imgs.items():
        width, height = img.size
        summary_image.paste(img, (curr_x, curr_y))
        curr_x += width + margin
    return summary_image


# Renderer for dictionary input with fields
def render_frames(sc2obs, components, margin=20):
    if isinstance(sc2obs, list) and ( isinstance(sc2obs[0], torch.Tensor) or\
                                     isinstance(sc2obs[0], np.ndarray)):
        # List of tensors, resulting from observation model
        #sc2obs = xhat2obs(sc2obs, components)
        sc2obs = ohat2sc2(sc2obs, components)
        return [_render(sc2o, components, margin=margin) for sc2o in sc2obs]
    elif isinstance(sc2obs, list):
        # List of NamedNumpyArrays
        return [_render(o, components, margin=margin) for o in sc2obs]
    else:
        return _render(sc2obs, components, margin=margin)
  

# Renderer for dictionary input with fields, and the array of original raw images for individual analyses
def render_frames2(sc2obs, components, margin=20, add_legend=True):
    img_dict = sc2obs['data']['feature_screen']
    name2imgs = {}
    summary_width = 0
    summary_height = 0
    img_by_component = {}
    for component in components:
        if hasattr(component, 'name'):
            field = component.name
        else:
            field = str(component)
        is_numeric = isinstance(component, NumericFrame) or str(component).endswith("ratio")
        if is_numeric:
            palette = NUMERIC_RANGE_COLORS
            is_numeric_key = True
        else:
            palette = CATEGORICAL_COLORS
            is_numeric_key=False
        arr = np.array(img_dict[field]) # This needs to be done as NamedNumpyArrays have issues with Numpy functions
        img = plot_categorical_map(arr, name=field, palette=palette, is_numeric_key=is_numeric_key, add_legend=add_legend)
        name2imgs[field] = img
        width, height = img.size
        summary_width += width + margin
        img_by_component[component.name] = img
        summary_height = max(height, summary_height)
    summary_image = Image.new("RGB", (summary_width, summary_height),(255,255,255))
    curr_x, curr_y = margin, 0
    for name, img in name2imgs.items():
        width, height = img.size
        summary_image.paste(img, (curr_x, curr_y))
        curr_x += width + margin
    return summary_image, img_by_component
        

