from .mc_box import MCBoxSpec
from .channels import SPACE_TYPE, ChannelSpec, DEFAULT_EMBEDDING_DIM


"""
This contains routines for working with different types of inputs and outputs.

The ConvVAE itself is spatial in nature, and contains a common encoder and decoder
architecture.  The decoder goes to a FC, which is then passed to multiple heads
for reconstruction, per MultiConvOutput).

For outputs that bypass this and go straight from the latent to the target, 
we are currently using MultiMLPOutput.  This is a MLP for now, since most
of these targets are nonspatial.
"""



class ShapedChannelSpec(ChannelSpec):
    """ A ChannelSpec that features a shape (instance-specific).
    The target is presumed to apply over the entire shape.
    Note: Does not feature a channel index, as the data is spread
    through the target shape.
    """
    def __init__(self, name, space_type,
                 shape,
                 low=None, high=None):
        super(ChannelSpec, self).__init__(name, space_type,
                                          low=low, high=high)
        self.shape = shape