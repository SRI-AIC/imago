from torch import nn as nn

from imago.spaces.channels import DEFAULT_EMBEDDING_DIM
from imago.spaces.mc_box import *

"""
This feeds a non-spatial Box into an input format suitable for the
MLP backbone.  This uses the space specifier to translate the inputs
into the following form,

   (batch, C)
   
Where C is the channel, with each feature arrayed in order.  

For categoricals, this will convert each into an embedding, which is
then concatenated onto the channel.
"""

class MultiMLPInput(nn.Module):
    def __init__(self, space_specifier:MCBoxSpec, device="cpu"):
        super(MultiMLPInput, self).__init__()
        self.device = device
        self.space_specifier = space_specifier
        self.embeddings = nn.ModuleDict()

        # Setup embeddings, indexing them by the name
        for channel_spec in self.space_specifier.channel_specs:
            if channel_spec.space_type == SPACE_TYPE.CATEGORICAL:
                space_size = channel_spec.size
                embedding_dim = DEFAULT_EMBEDDING_DIM
                self.embeddings[channel_spec.name] = nn.Embedding(space_size,
                                                                  embedding_dim).to(self.device)

    def forward(self, O):
        """
        Given an observation of the form (batch, F), F indexing the features, constructs
        and stacks the channel based input of form (batch, C) that will go into the
        MLPbackbone.  Each channel is readjusted as follows,
        - Categorical: Replaced with embedding value (stacked onto C)
        - Numeric: Rescaled value to 0-1 range
        :param O:
        :return:
        """
        batch_size = O.shape[0]
        O_p = [] # updated observation to pass to MLP backbone model
        for channel_spec in self.space_specifier.channel_specs:
            if channel_spec.is_numeric:
                A = self.space_specifier.get_channel_slice(O,
                           channel_spec.channel_idx).to(dtype=torch.float, device=self.device)
                A = channel_spec.norm(A)
                O_p.append(A)
            elif channel_spec.is_categorical:
                # Convert into the embedding
                C = self.space_specifier.get_channel_slice(O,
                               channel_spec.channel_idx).to(dtype=torch.long, device=self.device)
                A = self.embeddings[channel_spec.name](C)
                O_p.append(A)
            else:
                raise Exception("Unknown space type={}!".format(channel_spec.space_type))
        O_p = torch.cat(O_p, 1)
        return O_p