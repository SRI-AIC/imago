# Based on https://github.com/hujinsen/pytorch_VAE_CVAE/blob/master/CVAE.ipynb
from __future__ import print_function
import numpy as np
from PIL import Image, ImageDraw
from pysc2.lib.units import Neutral, Protoss, Terran, Zerg
# from pysc2.lib.units import get_unit_type
from xkcd_color_survey import colors as xkcd_colors
from sc2recorder.utils import ensure_numpy
from imago_prev.data.datasets import ALLEGIANCE_NAMES, split_alleg_unitid

import random
random.seed(133178)
filtered_xkcd_colors = [(colname, colvalue) for colname, colvalue in list(xkcd_colors.items()) \
          if not ("deep" in colname) and not ("burnt" in colname) and not ("black" in colname)]
random.shuffle(filtered_xkcd_colors)

marine_colors = [("Blue", "#0505ff"), ("Green", "#05ff05"), ("White", "#ffffff"), ("Red", "#ff0505")]

colors = marine_colors + filtered_xkcd_colors

def get_unit_type(raw_unit_id):
    # Adjust the UNIT ID to show allegiance
    allegiance_idx, unit_id = split_alleg_unitid(raw_unit_id)
    for race in (Neutral, Protoss, Terran, Zerg):
        try:
            return ALLEGIANCE_NAMES[allegiance_idx], str(race(unit_id))
        except ValueError:
            pass  # Wrong race.

class UnitTypeRenderer:
    """
    Rendering utility, drawing out color images of unit_types arrayed on a single 2D array.
    Note that this does not display allegiances.
    
    NOTE: This must be backed by an ObserverDataset, which pulls in lookups and formatting routines.
    """
    def __init__(self, metadata):
        self.metadata = metadata
        DISTINCT_UNITS = metadata.DISTINCT_UNITS
        N_DISTINCT_UNITS = metadata.N_DISTINCT_UNITS
        self.UNIT2COLOR = {}
        key_unit_width = 50
        key_img = Image.new("RGB", (800, N_DISTINCT_UNITS * key_unit_width), (255, 255, 255))
        key_drawer = ImageDraw.Draw(key_img)

        # Set up distinct units and assign RGB values for these
        key_y = 0
#        key_font = ImageFont.truetype("Courier", 12)
        for (cname, cvalue), uid in zip(colors, DISTINCT_UNITS):
            if uid != 0:
                allegiance_name, unit_name = get_unit_type(uid)
                print("{}\t{}-{}\t{}".format(uid, allegiance_name, unit_name, cname))
                self.UNIT2COLOR[uid] = cvalue
                key_drawer.rectangle([5, key_y, 25, key_y + key_unit_width], fill=cvalue)
                key_drawer.text([30, key_y], text=unit_name, fill=(0,0,0))
 #               key_drawer.text([30, key_y], text=unit_name, fill=(0,0,0), font=key_font)
                key_y += key_unit_width
        self.key_img = key_img

    def display(self, X, type="unit"):
        return self.unit_type_display(X)

    def unit_type_display(self, X, bgcolor=(0,0,0), disp_width=3):
        """
        Given a batch of 2D array of unit_types (int), displays the units according to the color key
        and returns the list of images.
        :param X:
        :param bgcolor:
        :param disp_width:
        :return:
        """
        # Displays SC2 data image
        X = ensure_numpy(X)
        if len(X.shape) == 3 and (np.all(np.unique(X) == [0, 1]) or \
                np.all(np.unique(X) == [0])):
            # In the case someone decided to take a raw one-hot (from the dataset) and
            # display it, unsqueeze and convert it to the proper form
            X = self.metadata.onehot2unit_type(np.expand_dims(X, 0))
        elif len(X.shape) == 4:
            # This is a raw one-hot, convert into thge unit type display using the
            # metadata
            X = self.metadata.onehot2unit_type(X)
        batch_size, height, width = X.shape
        imgs = []
        for bidx in range(batch_size):
            img = Image.new("RGB", (disp_width * width, disp_width * height), bgcolor)
            drawer = ImageDraw.Draw(img)
            for y in range(height):
                for x in range(width):
                    uid = X[bidx, y, x]
                    if uid > 0:
                        x1 = x * disp_width
                        x2 = x1 + disp_width
                        y1 = y * disp_width
                        y2 = y1 + disp_width
                        drawer.rectangle(((x1 , y1), (x2, y2)), fill=self.UNIT2COLOR[uid])
            imgs.append(img)
        return imgs
