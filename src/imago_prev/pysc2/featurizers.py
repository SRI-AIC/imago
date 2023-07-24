""" Featurization routines.  Take in observations and returns back dicts that can be
serialized via JSON.  Used with the WrapObserverAgent (observer.py)

Currently these require the following observations to be present:

feature_screen
"""

from pdb import set_trace
from collections import namedtuple, Counter
import numpy as np
from pysc2.lib import features
from sc2recorder.utils import ensure_numpy
import math

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

def get_valid_coords(M):
    """ Gets the X,Y coordinates of filtered 2D array input.  Note that rows index the y, cols index the x.
    This returns coordinates back in (x, y) form.
    """ 
    y, x = M.nonzero()  # equiv to np.where(M > 0)
    coords = []
    for idx in range(len(y)):
        coords.append((x[idx], y[idx]))
    return coords
      
CenterOfMass = namedtuple('CenterOfMass', ['mean_y', 'mean_x', 'std_y', 'std_x'])
    
    
def get_center_of_mass2(locs, width, height):
    if len(locs[0]) == 0:
        return CenterOfMass(0.5, 0.5, 0, 0)
    mean_x = float(np.mean(locs[1]) / width)
    mean_y = float(np.mean(locs[0]) / height)
    std_x = float(np.std(locs[1]) / width)
    std_y = float(np.std(locs[0]) / height)
    return CenterOfMass(mean_y, mean_x, std_y, std_x)

def featurize_rvf(obs, features=None):
    """
    Corrected featurizer, using the observation arrays instead of the raw_units
    information.  Currently we are getting odd inconsistencies between the listings
    in raw_units versus those in the arrays.
    :param obs:
    :param features:
    :return:
    """
    if features is None:
        features = {}
    FS = obs['data']['feature_screen']
    friendly_idxes = np.where(FS['player_relative'] == _PLAYER_SELF)
    enemy_idxes = np.where(FS['player_relative'] == _PLAYER_ENEMY)
    neutral_idxes = np.where(FS['player_relative'] == _PLAYER_NEUTRAL)
    height, width = FS['player_relative'].shape
    friendly_COM = get_center_of_mass2(friendly_idxes, width, height)
    bandits_COM = get_center_of_mass2(enemy_idxes, width, height)
    neutral_COM = get_center_of_mass2(neutral_idxes, width, height)
    features["center_of_mass"] = {"friendly": friendly_COM, "bandit": bandits_COM, "neutral": neutral_COM}
    # features["unit_type_hist"] = {"friendly": friendly_hist, "bandit": bandit_hist, "neutral": neutral_hist}
    return features



#
# BINARY FEATURES FOLLOW
#
from enum import IntEnum


class BinaryFeat(IntEnum):
    FRIENDLY_SMALL = 0
    FRIENDLY_MEDIUM = 1
    FRIENDLY_LARGE = 2
    ENEMY_SMALL = 3
    ENEMY_MEDIUM = 4
    ENEMY_LARGE = 5
    DISTANCE_CLOSE = 6
    DISTANCE_MEDIUM = 7
    DISTANCE_FAR = 8


def pprint_binaryfeat(c, ret_str=False):
    c = ensure_numpy(c)
    ret = ""
    for feat, c_val in zip(BinaryFeat, c):
        ret += "{}:\t{:.5f}\n".format(feat.name, c_val)
    if ret_str:
        return ret
    print(ret)

def force_size_offset(faction_idxes):
    """
    Returns small|med|large by offset
    NOTE: using a 64x64 grid, each Marine unit seems to take up
    a 2x2 spot in the semantic unit listing.  For now, we'll treat this as
    'area' and work from there.
    :param faction_idxes:
    :return:
    """
    if len(faction_idxes[0]) <= 4 * 3:
        return 0
    if len(faction_idxes[0]) <= 4 * 8:
        return 1
    return 2


def force_distance(com1, com2):
    dist_x = pow(com1.mean_x - com2.mean_x, 2)
    dist_y = pow(com1.mean_y - com2.mean_y, 2)
    distance = math.sqrt(dist_x + dist_y)
    if distance <= 0.3:
        return 0
    if distance <= 0.6:
        return 1
    return 2


def featurize_binary(obs, wt_denom=1.0):
    """
    Generates binary attribute features, as opposed to real valued ones.
    :param obs:
    :param features:
    :return:
    """
    features = np.zeros(len(BinaryFeat))
    FS = obs['data']['feature_screen']
    friendly_idxes = np.where(FS['player_relative'] == _PLAYER_SELF)
    enemy_idxes = np.where(FS['player_relative'] == _PLAYER_ENEMY)
    height, width = FS['player_relative'].shape
    friendly_COM = get_center_of_mass2(friendly_idxes, width, height)
    enemy_COM = get_center_of_mass2(enemy_idxes, width, height)
    friendly_size_offset = force_size_offset(friendly_idxes)
    enemy_size_offset = force_size_offset(enemy_idxes)
    force_distance_offset = force_distance(friendly_COM, enemy_COM)
    features[BinaryFeat.FRIENDLY_SMALL + friendly_size_offset] = 1
    features[BinaryFeat.ENEMY_SMALL + enemy_size_offset] = 1
    features[BinaryFeat.DISTANCE_CLOSE + force_distance_offset] = 1
    features = features / wt_denom
    return features
