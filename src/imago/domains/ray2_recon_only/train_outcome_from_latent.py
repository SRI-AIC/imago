"""
This takes a PVaeModel that has been trained and
models it with a new set of targets.  The intent is to allow
a reconstruction only model to be developed, and then train an
outcome model just on those latents.

This uses the model trained from the recon only, and uses the full model
scaffolding to train, but freezes the gradient on the recon and VAE component.

The resulting model should be directly usable in our usual experiment scaffolds.
"""
from pathlib import Path
import collections
from torch.utils.data import DataLoader
import torch.nn as nn

from imago import MODELS_ROOT, CAML_ROOT
from imago.domains.ray2_recon_only.ray2_ro_setup import load_model
from imago.spaces.outputs.mlp_output import MLPOutput
from imago.domains.reaverassaulty2.ray2_setup import LATENT2TARGET_SPECS, mcbox_spec, load_data
from imago.outcome_only_trainer import train

DEBUG=False
DEVICE="cuda:0"
RO_MODELS_DIR = Path(MODELS_ROOT, "ray2_recon_only")

RECON_MODEL = Path(RO_MODELS_DIR, "RAY2_RO_lr1en3_Wrecon10", "model.pt")
model = load_model(RECON_MODEL, device=DEVICE, force_new=False)

for p in model.vae_model.parameters():
    p.requires_grad = False

# Now graft in modalities
model.latent2target_specs = { spec.name: spec for spec in LATENT2TARGET_SPECS }
l2ts = collections.OrderedDict()
for spec in model.latent2target_specs.values():
    if "value" in spec.name:
        print("Using block 2 for {}".format(spec))
        use_block2 = True
    else:
        use_block2 = False
    l2ts[spec.name] = MLPOutput(spec, model.vae_model.N_LATENT,
                                device=DEVICE,
                                use_block2=use_block2)  # Test to see impact on value function
model.latent2targets = nn.ModuleDict(l2ts)
OUTCOME_FROM_LATENT_MODEL_FPATH = Path(CAML_ROOT, "output", "outcome4latent")
model.force_deterministic = True

BATCH_SIZE=16
train_ds, test_ds = load_data(load_debug=DEBUG)
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

train(model, OUTCOME_FROM_LATENT_MODEL_FPATH, mcbox_spec,
      train_dataloader, test_dataloader,
      device=DEVICE, epochs=10000, lr=1e-3,
      step_limit=16395696
      )
