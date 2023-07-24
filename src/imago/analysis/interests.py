import torch
from torch.distributions import Categorical
import pdb

from imago.utils import *

"""
Postprocessing to obtain derived interestingness measures from a given model.
"""

def get_interests(That,
                  interests_profile):
    """
    Generate interestingness values
    :param:interests_profile: Details for how to access the behavior and value offsets in the latent targets

    Note: Action entropies are concatenated together, in order
    """
    action_entropies = []
    action_idxes = interests_profile.action_idxes
    value_idx = interests_profile.value_idx
    for act_idx in action_idxes:
        Ad = That[act_idx]
        pA = Categorical(Ad)
        entr_pA = pA.entropy()
        action_entropies.append(entr_pA)
    iV = That[value_idx]
    action_entropies = torch.stack(action_entropies, dim=1)
    return action_entropies, iV


def getat_Z(Zp, model, interests_profile):
    Zp = ensure_torch(model.device, Zp)
    Ohat, That = model.forward_Z(Zp)
    entr_pA, iV = get_interests(That, interests_profile)
    return Ohat, That, entr_pA, iV


class InterestsProfile:
    """ The offsets to the action and value indices for this type of
    Interestingness variables.

    TODO: Cleanly abstract this later on.
    """
    def __init__(self, action_idxes, value_idx):
        self.action_idxes = action_idxes
        self.value_idx = value_idx

    def __str__(self):
        return "ValueIdx={}  ActionIdxes={}".format(self.value_idx,
                                                    ", ".join([str(x) for x in self.action_idxes]))
