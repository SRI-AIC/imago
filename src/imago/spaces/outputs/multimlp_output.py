import collections

import torch.nn as nn

from imago.spaces.outputs.multihead_output import SpatialMultiHeadOutput
from imago.spaces.outputs.mlp_output import MLPOutput
from imago.spaces.mc_box import *

class MultiMLPOutput(SpatialMultiHeadOutput):
    """
    Represents a cluster of targets that are not spatially oriented (MLP).

    Note: Although this derives from a spatially oriented class, the shape
    here really is (1,), in order to allow cross compatibility with
    loss code
    """

    def __init__(self, space_specifier:MCBoxSpec,
                 device="cpu"):
        super(MultiMLPOutput, self).__init__(space_specifier,
                                             device=device)

    def _init_heads(self):
        heads = collections.OrderedDict()
        for spec in self.space_specifier.channel_specs:
            if "value" in spec.name:
                print("Using block 2 for {}".format(spec))
                use_block2 = True
            else:
                use_block2 = False
            heads[spec.name] = MLPOutput(spec, self.vae_model.N_LATENT,
                                                 device=self.device,
                                                 use_block2=use_block2) # Test to see impact on value function
        self.heads = nn.ModuleDict(heads)
