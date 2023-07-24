"""
Joined recon and outcome model, set up so we can use same experimental scaffolding
with double model
"""
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

from imago.domains.ray2_recon_only.ray2_ro_setup import load_model as load_recon_model

class JoinedModel(nn.Module):
    def __init__(self,
                 space_specifier,
                 latent2target_specs=[],  # List of additional specifiers, for going from latent to target
                 N_LATENT=256,
                 device="cpu"):
        super(JoinedModel, self).__init__()
        self.device = device
        self.space_specifier = space_specifier
        self.latent2target_specs = { spec.name: spec for spec in latent2target_specs }
        self.recon_model = load_recon_model(device=device)
        # self.outcome_model =