from gym.spaces.box import Box
import torch
import numpy as np

from imago.spaces.channels import SPACE_TYPE
from imago.utils import ensure_torch

"""
Extensions for working with Spaces as reconstructable observations

TODO: Extend to spaces.Dict, e.g., from Cameleon,

        self.observation_space = spaces.Box(
                                    low=0,
                                    high=255,
                                    shape=(width, height, 3),
                                    dtype='uint8')
        self.observation_space = spaces.Dict({
            'image': self.observation_space})

TODO: Include code that converts a spaces.Dict into a batched observation.  We may
have to consider whether or not to have separate encoding piplines for them, as opposed
to attempting to integrate them all together.
"""


class MCBoxSpec():
    """
    MultiChannel Box Specifier
    This holds information about how to reconstruct a given Box space, where each of the channels
    can be multiple types (categorical, numeric, sparse numeric).
    Input adjustment:
    - Numeric values are treated as is, as long as (H, W, ...) is maintained.  Values are rescaled according to the
      low/high values, so we get a 0-1 range.  This includes multiple channel numerics (RGB)
    - Categoricals: Inputs get mapped to an embedding layer, and are a multinomial target on reconstruction.

    NOTE: This assumes all of the input channels shares this space.
    """
    def __init__(self, channel_specs, space, channel_axis=-1):
        """
        channel_axis: Axis position to data channels in the original space shape
        """
        super(MCBoxSpec, self).__init__()
        assert(isinstance(space, Box))
        self.channel_specs = channel_specs
        self.channel_lookup = {
            cspec.name: cspec for cspec in self.channel_specs
            }
        self.space = space  
        self.channel_axis = channel_axis  # Index into axis where channel information is retained.

        # If -1 is used, get the index of the last 
        if self.channel_axis == -1:
            self.channel_axis = len(self.space.shape) - 1

        # For channel specs whose low/high are not specified, use the space's
        # default.  For position, default to the offset in the list.
        for cidx, channel_spec in enumerate(self.channel_specs):
            if channel_spec.low is None:
                channel_spec.low = self.space.low
            if channel_spec.high is None:
                channel_spec.high = self.space.high
            if channel_spec.channel_idx is None:
                channel_spec.channel_idx = cidx

    def __iter__(self):
        """ Returns a generator returning (space, space_name, space_type) for contained spaces"""
        return self.channel_specs

    def get_channel_slice(self, O, cidx, batch_idx=None):
        """
        Given a Box space with a single data channel axis, returns the sliced
        of the data at the cidx offset. Note: This expects O to have a spatial dimension already attached
        :param O:
        :param cidx:
        :return:
        """
        if isinstance(cidx, str):
            cidx = self.channel_lookup[cidx].channel_idx
        slice_list = [slice(None)] * len(O.shape)
        if self.channel_axis + 1 > len(slice_list):
            raise Exception("Slice list len={} for shape does not match observation dims={}, attempting to access channel axis (plus batch idx) {}.  If this is a non-spatial input, were false spatial dims added?".format(len(slice_list), O.shape, self.channel_axis + 1))
        slice_list[self.channel_axis + 1] = cidx # Need to account for batch idx
        if batch_idx is not None:
            slice_list[0] = batch_idx
        return O[tuple(slice_list)]

    def get_spatial_dims(self):
        """ Gets the batch and spatial dims in sequence for a Box covered by
        this MultiChannel Box Specifier.  If none present, e.g., Box shape (N,), returns empty."""
        return [x for idx, x in enumerate(self.space.shape) if idx != self.channel_axis]    

    def dict2X(self, X_dict):
        """
        Given a dictionary with channel names to the channel data Tensors,
        restacks them into the Box space, with data slices along the
        channel_axis.

        NOTE: This will destroy any information needed for computing losses.
        
        For categorical channels, we apply a softmax first, where
        the shape is (batch, labels, height, width)
        :param X_dict:
        :return:
        """
        L = []
        for channel_spec in self.channel_specs:
            X = X_dict[channel_spec.name]
            if channel_spec.is_numeric:
                X = X.squeeze(1)
                X = channel_spec.denorm(X) 
            elif channel_spec.is_distributional:
                X = X.squeeze(1)
            elif channel_spec.space_type == SPACE_TYPE.STANDARDIZED:
                X = X.squeeze(1)
            elif channel_spec.space_type == SPACE_TYPE.CATEGORICAL:
                # Cast to same type so stacking works correctly
                X = torch.argmax(X, dim=1, keepdim=False).to(torch.float32)
            elif channel_spec.space_type == SPACE_TYPE.BINARY:
                Xp = torch.zeros(X.shape).to(X.device)
                Xp[X > 0.5] = 1
                X = Xp.squeeze(1)
            else:
                raise Exception("Unsupported space type={}".format(channel_spec.space_type))
            L.append(X)
        X =  torch.stack(L, dim=(self.channel_axis+1))
        assert all([X.shape[i+1] == self.space.shape[i] for i in range(len(self.space.shape))])
        return X

    def X2dict(self, X):
        """
        Expands the given tensor into dict where the channel name references
        the data slice.  Used for piecemeal examination, such as computing
        individual losses.

        TODO: Allow option to expand channel into one-hot versions
        :param X:
        :return:
        """
        return {
            channel_spec.name: self.get_channel_slice(X, channel_spec.channel_idx)
            for cidx, channel_spec in enumerate(self.channel_specs)
        }

    def make_onehot(self, X, device="cpu"):
        """
        Given the tensor, constructs a one-hot capable tensor for comparison.
        This applies normalization as needed for numeric channels.
        :param X:
        :return:
        """
        X = ensure_torch(device, X)
        spatial_dims = self.get_spatial_dims()
        C = []
        for channel_spec in self.channel_specs:
            c = self.get_channel_slice(X, channel_spec.channel_idx)
            if channel_spec.space_type == SPACE_TYPE.CATEGORICAL:
                c = c.unsqueeze(self.channel_axis + 1)
                c1 = torch.zeros(c.shape[0], channel_spec.size, *spatial_dims).to(X.device)
                c1 = c1.scatter_(1, c.to(dtype=torch.int64), 1)
                C.append(c1)
            else:
                c = channel_spec.norm(c)  # Allow comparability with 0/1
                C.append(c.unsqueeze(self.channel_axis + 1))
        C = torch.cat(C, dim=1)
        return C


    def compute_Odiff(self, O1, O2, device="cpu"):
        O1 = self.make_onehot(O1, device=device)
        O2 = self.make_onehot(O2, device=device)
        return torch.sum(O1 - O2).item()



