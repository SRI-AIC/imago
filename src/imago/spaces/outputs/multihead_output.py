import torch.nn as nn
import torch.nn.functional as F

from enum import IntEnum

from imago.layers import GatedConvRegressionHead
import imago.losses as losses
from imago.spaces.mc_box import *
import pdb


class OUTPUT_TYPE(IntEnum):
    EMBED2CONV = 0 # Conv embedding -> Convolutional
    LATENT2MLP = 1 # Latent code -> MLP

def multinomial_dice_loss(X, Y):
    """
    Scores the DICE loss according to the one-hot activations
    for each of the targets. For targets that are
    :param X:
    :param Y:
    :return:
    """


class SpatialMultiHeadOutput(nn.Module):
    """
    Base class for deriving multiple outputs.

    This is arranged so a VAE backbone will generate a (B, C, H, W)
    decoded representation (Torch order).  Individual targets will then use the C channels
    and perform their own interpretation.

    """
    def __init__(self, space_specifier: MCBoxSpec,
                 decoder_embedding_size=10,  # Embedding size for the decoder, prior to processing by the heads
                 device="cpu"):
        """ Given a MultiTuple specification, constructs the target heads and losses used to
        reconstruct and train these components.

        This focuses primarily on spatial targets, and reconstructs from the VAE decoder.
        """
        super(SpatialMultiHeadOutput, self).__init__()
        self.device = device
        self.numeric_c_dim = decoder_embedding_size # Target output channel size, for use by spatial decoders
        self.space_specifier = space_specifier
        self.num_output_channels = 1  # TODO: Infer from the shape of the tuple specs

        # List of decoder embedding to targets, for each channel
        self._init_heads()

    def __str__(self):
        ret = []
        for head, channel_spec in zip(self.heads, self.space_specifier.channel_specs):
            ret.append("{}/{}: module={}".format(channel_spec.name,
                                                 channel_spec.space_type,
                                                 str(head).replace('\n', ', ')))
        return "\n\n".join(ret)

    def _init_heads(self):
        return NotImplementedError()

    def dim(self):
        return self.numeric_c_dim

    def forward(self, X):
        """ Given the decoder output, a map of shape (batch, C, H, W), performs forward inference over each
        of the heads and returns a dictionary of outputs for each, where each output is indexed by name.  The
        batch form is retained for each of the outputs.

        This allows for each output to be selectively checked for losses, and examined separately.
        TODO: Reconstruction of the Box specifier
        """
        outputs = {}
        for head, channel_spec in zip(self.heads, self.space_specifier.channel_specs):
            outputs[channel_spec.name] = head(X)
        return outputs

    def recon_loss(self, X_guess, O_gold):
        """ Given the guess produced by forward with ret_training set to True (in order to
        return the dictionary view of each semantic frame), the Observation tensor passed in,
        computes the reconstruction losses.

        Note we have to explode a Box into target tensors by constituent channels,
        as they may have entirely different objectives.
        """
        loss = 0.
        batch_size = len(O_gold)
        dims = self.space_specifier.get_spatial_dims()
        dim_h = dims[0]
        dim_w = dims[1]
        #pdb.set_trace()
        #X_guess = self.space_specifier.X2dict(X_guess)

        for cidx, (head, channel_spec) in enumerate(zip(self.heads, self.space_specifier.channel_specs)):
            space_name = channel_spec.name
            space_type = channel_spec.space_type

            Xhat = X_guess[space_name]
            Y = self.space_specifier.get_channel_slice(O_gold, cidx)
            if space_type == SPACE_TYPE.CATEGORICAL:
                hyp = F.log_softmax(Xhat, dim=1)
                loss += F.nll_loss(hyp, Y.to(dtype=torch.long,
                                             device=self.device))  # Remove last so we get H,W,target
            elif space_type == SPACE_TYPE.BINARY:
                target = Y.to(dtype=torch.float, device=self.device)
                loss += F.binary_cross_entropy(Xhat.squeeze(1), target)
            elif space_type == SPACE_TYPE.NUMERIC:
                target = Y.to(dtype=torch.float, device=self.device)
                target = channel_spec.norm(target)
                # Make targets match PyTorch's (B,C,H,W) format, so targets are lined
                # up.
                target = target.view(batch_size, -1, dim_h, dim_w)
                # Squeezing the target, as the target has the extra dimension for the value
                loss += F.mse_loss(Xhat, target)
            elif space_type == SPACE_TYPE.SPARSE_NUMERIC:
                # Do a two stage check.  First treat it as a 0-1 problem, then
                # regress the value just on the mask of guessed values.
                # We squeeze the channel axis, as in the 2D case this will always be
                # a single value.
                H1 = head.last_gate.squeeze(1)
                T1 = Y.to(self.device, dtype=torch.float)

                # TODO: Dynamically select the correct channel axis
                if self.space_specifier.channel_axis == 0:
                    T1[torch.where(O_gold[:, cidx, ...] > 0)] = 1
                elif self.space_specifier.channel_axis == 2:
                    T1[torch.where(O_gold[:, :, :, cidx] > 0)] = 1
                else:
                    raise Exception("Error: Shape and channel axis not supported yet")
                loss1 = F.binary_cross_entropy(H1, T1)
                loss += loss1

                # 2nd stage, mask against 0/1 targets in stage 1 and regress
                # Again, squeeze the channel axis of the gated regression hypothesis
                # as we only have a single numeric axis
                # Note we mask against the gold mask, as other values are zeroed and we
                # do not want to sparsify against any false positives on the mask.
                H2 = head.last_guess.squeeze(1) * T1
                T2 = Y.to(dtype=torch.float, device=self.device)
                T2 = channel_spec.norm(T2)
                loss2 = F.mse_loss(H2, T2.squeeze(1))
                loss += loss2
            elif channel_spec.is_distributional:
                target = Y.to(dtype=torch.float, device=self.device)
                target = channel_spec.norm(target)
                if len(target.shape) == 3:
                    target = target.unsqueeze(-1)
                else:
                    target = target.view(batch_size, dim_h, dim_w, -1)
                if space_type == SPACE_TYPE.JSD:
                    divg_fn = losses.JSD
                elif space_type == SPACE_TYPE.DIST_KL_DIV:
                    divg_fn = losses.KLDIV
                else:
                    raise Exception("Unsupported distribution space type={}".format(space_type))
                loss += losses.dist_divg_fn(Xhat, target.squeeze(1), divg_fn)
            else:
                raise Exception("Unsupported channel type={}".format(channel_spec.space_type))
        return loss

