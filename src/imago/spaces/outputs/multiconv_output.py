import torch.nn as nn

from imago.layers import GatedConvRegressionHead
from imago.spaces.outputs.multihead_output import SpatialMultiHeadOutput
from imago.spaces.mc_box import *


class MultiConvOutput(SpatialMultiHeadOutput):
    """
    Spatially significant convolution based decoder
    """
    def __init__(self, space_specifier: MCBoxSpec,
                 decoder_embedding_size=10,  # Embedding size for the decoder, prior to processing by the heads
                 device="cpu"):
        """ Given a MultiTuple specification, constructs the target heads and losses used to
        reconstruct and train these components.

        This focuses primarily on spatial targets, and reconstructs from the VAE decoder.
        """
        super(MultiConvOutput, self).__init__(space_specifier,
                                              decoder_embedding_size=decoder_embedding_size,
                                              device=device)

    def _init_heads(self):
        self.heads = nn.ModuleList()
        for channel_spec in self.space_specifier.channel_specs:
            if channel_spec.space_type == SPACE_TYPE.NUMERIC:
                head = nn.Sequential(nn.ConvTranspose2d(self.numeric_c_dim,
                                                        self.num_output_channels,
                                                        kernel_size=1, stride=1),
                                     nn.Sigmoid()).to(self.device)
            elif channel_spec.is_distributional:
                head = nn.Sequential(nn.ConvTranspose2d(self.numeric_c_dim,
                                                        self.num_output_channels,
                                                        kernel_size=1, stride=1),
                                     nn.Sigmoid()).to(self.device)
            elif channel_spec.space_type == SPACE_TYPE.SPARSE_NUMERIC:
                head = GatedConvRegressionHead(self.numeric_c_dim).to(self.device)
            elif channel_spec.space_type == SPACE_TYPE.CATEGORICAL:
                space_vocab_size = channel_spec.size
                head = nn.Sequential(nn.ConvTranspose2d(self.numeric_c_dim,
                                                        space_vocab_size,
                                                        kernel_size=1, stride=1),
                                     nn.ReLU()).to(self.device)
            elif channel_spec.space_type == SPACE_TYPE.BINARY:
                space_vocab_size = channel_spec.size
                head = nn.Sequential(nn.ConvTranspose2d(self.numeric_c_dim,
                                                        space_vocab_size,
                                                        kernel_size=1, stride=1),
                                     nn.Sigmoid()).to(self.device)
            else:
                raise Exception("Unknown space type={}".format(channel_spec.space_type))
            self.heads.append(head)

