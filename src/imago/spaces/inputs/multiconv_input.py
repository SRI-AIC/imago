from torch import nn as nn

from imago.spaces.channels import DEFAULT_EMBEDDING_DIM
from imago.spaces.mc_box import *

"""
This feeds a spatially arranged Box input into a convolutional encoder.  This uses
the space specifier (currently MCBoxSpec) to muster inputs into a 2D 
convolutional form.  This includes unpacking of categoricals into embeddings.
All of the features are inserted into the channel portion of the inputs, which
should then be combined with a MultiConvVAE backbone.

This expects input observations to be a single tensor, of the form

   (batch, H, W, F)
   
Where F indexes the feature, H and W are 2d coordinates.

For categorical, we use just the first channel, which contains longs that index 
which label is present at that coordinate.  Categoricals are converted into
an embedding form, and this is stacked onto the input channel.
"""

class MultiConvInput(nn.Module):
    """
    Given a MultiTuple specification, sets up a network that takes a single observation tensor where the
    inputs are stacked together.

    Currently we only support Boxes, via BoxSpecifier, of shape and order (H, W, C)
    """
    def __init__(self, space_specifier:MCBoxSpec, device="cpu"):
        super(MultiConvInput, self).__init__()
        self.device = device
        self.space_specifier = space_specifier
        self.embeddings = nn.ModuleDict()

        # Compute embeddings for categoricals, indexing them by the name
        for channel_spec in self.space_specifier.channel_specs:
            if channel_spec.space_type == SPACE_TYPE.CATEGORICAL:
                space_size = channel_spec.size
                embedding_dim = DEFAULT_EMBEDDING_DIM
                self.embeddings[channel_spec.name] = nn.Embedding(space_size,
                                                                  embedding_dim).to(self.device)

    def dim(self):
        """
        Computes the total dimension for all of the inputs, stacked together.  Used primarily for
        computing the input to the ConvVAE
        """
        total = 0
        for channel_spec in self.space_specifier.channel_specs:
            if channel_spec.space_type == SPACE_TYPE.CATEGORICAL:
                total += self.embeddings[channel_spec.name].embedding_dim
            elif channel_spec.space_type == SPACE_TYPE.NUMERIC or \
                    channel_spec.space_type == SPACE_TYPE.SPARSE_NUMERIC or \
                    channel_spec.space_type == SPACE_TYPE.BINARY:
                total += 1
            else:
                raise Exception("Unsupported Space type={}".format(channel_spec.space_type))
        return total

    def forward(self, O):
        """ Given an observation tensor for the Box, formatted as (batch, H, W, C), constructs and stacks together the inputs
        proper that will go into the VAE.

        Each channel is adjusted as follows,
        - Categorical: Replaced with embedding value
        - Numeric: Rescaled value to 0-1 range

        Returns the formatted input, in Pytorch order (batch, C, H, W), ready for convolutional operations.
        """
        batch_size = O.shape[0]
        shape_tuple = self.space_specifier.get_spatial_dims()
        dim_h, dim_w = shape_tuple[0], shape_tuple[1]
        O_p = [] # Updated observation tensor to pass to model
        for channel_spec in self.space_specifier.channel_specs:
            if channel_spec.space_type == SPACE_TYPE.NUMERIC or \
                            channel_spec.space_type == SPACE_TYPE.SPARSE_NUMERIC:
                # Rescale and then shift so space is 0-1
                A = self.space_specifier.get_channel_slice(O, channel_spec.channel_idx).to(dtype=torch.float,
                                                                                           device=self.device)
                A = channel_spec.norm(A)
                if len(A.shape) == 3:
                    A = A.unsqueeze(-1)  # Unpack the data into its own channel layer
                else:
                    A = A.view(batch_size, dim_h, dim_w, -1) # Stack into a (b, h, w, c) shaped array
                O_p.append(A)
            elif channel_spec.space_type == SPACE_TYPE.CATEGORICAL:
                # Convert into embedding
                C = self.space_specifier.get_channel_slice(O, channel_spec.channel_idx).to(dtype=torch.long,
                                                                                           device=self.device)
                A = self.embeddings[channel_spec.name](C)  # Get embedding for each label
                O_p.append(A)
            elif channel_spec.space_type == SPACE_TYPE.BINARY:
                C = self.space_specifier.get_channel_slice(O, channel_spec.channel_idx).to(dtype=torch.float, device=self.device)
                if len(C.shape) == 3:
                    C = C.unsqueeze(-1) # Unpack channel layer
                else:
                    C = C.view(batch_size, dim_h, dim_w, -1) # Stack into channel end pre-stack norm
                O_p.append(C)
            else:
                raise Exception("Unknown space type={}!".format(channel_spec.space_type))
        O_p = torch.cat(O_p, 3).permute(0,3,1,2) # Permute into BxCxHxW, so we match Torch's ordering (for conv funcs)
        return O_p
