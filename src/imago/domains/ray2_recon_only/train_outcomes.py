"""
Trains the reconstruction only network from the semantic frame
"""
import os
from torch.utils.data import DataLoader

from imago import CAML_ROOT
from imago.domains.reaverassaulty2.ray2_setup import LATENT2TARGET_SPECS, load_data, mcbox_spec
from imago.models.outcomes_only import PVaeEncoders
from imago.outcome_only_trainer import train

DEVICE="cuda:0"
OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/outcomes/RAY2_RO_outcomes")
BATCH_SIZE=16

print("Loading data")

train_ds, test_ds = load_data(load_debug=False)
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

print("Loading model")

model = PVaeEncoders(mcbox_spec, latent2target_specs=LATENT2TARGET_SPECS,
                      device=DEVICE).to(DEVICE)
model.train()
train(model, OUTPUT_ROOT, mcbox_spec, train_dataloader, test_dataloader,
      device=DEVICE, epochs=10000, lr=1e-3,
      step_limit=16395696)  # Added in step limit for replicability