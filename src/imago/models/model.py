import torch.nn as nn
import collections
from enum import Enum

from imago.utils import ensure_torch
from imago.models.conv_vae import MultiInputConvVAE
from imago.models.mlp_vae import MLPVAE
from imago.spaces.inputs.multimlp_input import MultiMLPInput
from imago.spaces.inputs.multiconv_input import MultiConvInput
from imago.spaces.outputs.mlp_output import MLPOutput
from imago.spaces.outputs.multiconv_output import MultiConvOutput
from imago.spaces.outputs.multihead_output import SpatialMultiHeadOutput

class ModelType(Enum):
    CONV = 1
    MLP = 2

class PVaeModel(nn.Module):
    """
    Multiple heads map observations into a single (B, C, H, W) representation that is usable
    by the Torch's convolutional functions.  The VAE converts these into a decoded representation, C,
    and the heads each take this representation to generate their own outputs.
    """
    def __init__(self,
                 space_specifier,
                 latent2target_specs=[],  # List of additional specifiers, for going from latent to target
                 N_LATENT=256,
                 device="cpu",
                 force_deterministic=False):
        super(PVaeModel, self).__init__()
        self.device = device
        self.space_specifier = space_specifier
        self.latent2target_specs = { spec.name: spec for spec in latent2target_specs }

        # Specify the inputs and outputs for the convolutional component.
        # These share a fused encoder and decoder, with individual heads
        # performing the channel specific decoding after the decoder generated FC
        self.mt_in = MultiConvInput(space_specifier, device=device)
        self.mt_out = MultiConvOutput(space_specifier, device=device)

        # Assume the non-channel dims are (batch, height, width)
        dims = space_specifier.get_spatial_dims()
        assert len(dims) == 2
        height = dims[0]
        width = dims[1]

        # Currently we use a spatial model for all types. For domains
        # that are just real valued vectors, we use a spatial dim of (1, 1)
        self.vae_model = MultiInputConvVAE(self.mt_in.dim(),
                                           self.mt_out.dim(),
                                           height=height,
                                           width=width,
                                           N_LATENT=N_LATENT)
        # Now add the additional targets.
        # Each of these has their own processing network, with the
        # option of having multiple channels share the same MLP.
        l2ts = collections.OrderedDict()
        for spec in self.latent2target_specs.values():
            if "value" in spec.name:
                print("Using block 2 for {}".format(spec))
                use_block2 = True
            else:
                use_block2 = False
            l2ts[spec.name] = MLPOutput(spec, self.vae_model.N_LATENT,
                                                 device=device,
                                                 use_block2=use_block2) # Test to see impact on value function
        self.latent2targets = nn.ModuleDict(l2ts)
        self.to(device)
        self.force_deterministic = force_deterministic

    def forward(self, O, deterministic=False, ret_for_training=False):
        """
        :param O:
        :param deterministic: If True, avoids resampling and uses mean
        :param ret_for_training: If True, returns dict of channel names to slices, in original
                                format suitable for training. otherwise attempts to recreate
                                original input as much as possible.
        :return:
        """
        # Determine if we should do resampling
        deterministic = deterministic or self.force_deterministic
        x1 = self.mt_in(O)
        y1, z_mu, z_logvar, z = self.vae_model(x1, deterministic=deterministic)
        x2_dict = self.mt_out(y1)  # Reconstruction ordered by the space name, so we can use individual losses
        # Now get the latent2target outputs, generating them from the latent
        # and returning them as a dict.
        That = collections.OrderedDict()
        for name, l2t in self.latent2targets.items():
            That[name] = l2t(z)

        # Return dict, with original version
        if ret_for_training:
            return x2_dict, z_mu, z_logvar, z, That
        # Returns back in the same format as the original, including denormalizations
        # of the target variables
        ret_That = {}
        for k, v in That.items():
            target_spec = self.latent2target_specs[k]
            ret_That[k] = target_spec.denorm(v)

        # Note that space specifiers and targets share much the same routines, and
        # will be unified
        return self.space_specifier.dict2X(x2_dict), z_mu, z_logvar, z, ret_That

    def forward_Z(self, z, ret_for_training=False):
        z = ensure_torch(self.device, z)
        y1 = self.vae_model.forward_Z(z)
        y2 = self.mt_out(y1)
        That = collections.OrderedDict()
        for name, l2t in self.latent2targets.items():
            That[name] = l2t(z)
        if ret_for_training:
            return y2, That
        # Return original tensor, including denormalizing values
        ret_That = {}
        for k, v in That.items():
            target_spec = self.latent2target_specs[k]
            ret_That[k] = target_spec.denorm(v)
        return self.space_specifier.dict2X(y2), ret_That

    def recon_loss(self, X_guess:dict, O_gold):
        """ Gets recon loss, using the y2 dictionary returned by forward"""
        return self.mt_out.recon_loss(X_guess, O_gold)

    def latent2target_loss(self, That, Ts):
        """ Returns the loss for each of the latent2targets.
        Returns the (loss, OrderedDict(channel name -> loss))
        Expects the That and Ts to both be dicts."""

        # TODO:Assert keys align
        loss = 0.
        T_losses = collections.OrderedDict()
        for name, l2t in self.latent2targets.items():
            t_loss = l2t.loss(That[name], Ts[name])
            loss += t_loss
            T_losses[l2t.space_specifier.name] = t_loss.item()
        return loss, T_losses

