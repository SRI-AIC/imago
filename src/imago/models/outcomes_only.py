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
from imago.models.model import ModelType

class PVaeEncoders(nn.Module):
    """
    Encoder only, for re-use in alternate tests.  Emulates PVAEModel, but
    is set to emit only the targets

    TODO: Instead of a shared input conv, should be independent targets
    """
    def __init__(self,
                 space_specifier,
                 latent2target_specs=[],  # List of additional specifiers, for going from latent to target
                 N_LATENT=256,
                 device="cpu"):
        super(PVaeEncoders, self).__init__()
        self.device = device
        self.space_specifier = space_specifier
        self.latent2target_specs = { spec.name: spec for spec in latent2target_specs }
        self.mt_in = MultiConvInput(space_specifier, device=device)
        self.mt_out = MultiConvOutput(space_specifier, device=device)
        # Assume the non-channel dims are (batch, height, width)
        dims = space_specifier.get_spatial_dims()
        assert len(dims) == 2

        # Assume the non-channel dims are (batch, height, width)
        dims = space_specifier.get_spatial_dims()
        assert len(dims) == 2
        height = dims[0]
        width = dims[1]

        fc_in_dim = self.mt_in.dim() * height * width

        # Emulate the variational latent and just go as a FC
        self.fc = nn.Linear(fc_in_dim, N_LATENT)

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
            l2ts[spec.name] = MLPOutput(spec, N_LATENT,
                                        device=device,
                                        use_block2=use_block2)  # Test to see impact on value function
        self.latent2targets = nn.ModuleDict(l2ts)
        self.to(device)

    def forward(self, O, ret_for_training=False, deterministic=True):
        x1 = self.mt_in(O)
        batch_size = x1.shape[0]
        z = self.fc(x1.reshape(batch_size, -1))

        That = collections.OrderedDict()
        for name, l2t in self.latent2targets.items():
            That[name] = l2t(z)

        # Returns none for the Xhat and Z returns
        if ret_for_training:
            return None, None, None, None, That
        else:
            ret_That = {}
            for k, v in That.items():
                target_spec = self.latent2target_specs[k]
                ret_That[k] = target_spec.denorm(v)
            return None, None, None, None, ret_That

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