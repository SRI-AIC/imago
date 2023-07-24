"""
Usage:
cd caml
python -m pvae.datasets.cartpole.train_cartpole
"""

import sys
import torch
torch.autograd.set_detect_anomaly(True)

from torch.utils.data import DataLoader
from imago.domains.cartpole.cartpole_setup import *

from imago import CAML_ROOT
from imago.models.model import PVaeModel
from imago.trainer import train

from absl import logging
logging.get_absl_handler().python_handler.stream = sys.stdout

DEVICE = "cuda:0"
OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/pvae/cartpole_spatial_v2")

BATCH_SIZE=1500
train_dataset, test_dataset = load_data(add_spatial_dims=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = PVaeModel(mcbox_spec, latent2target_specs=LATENT2TARGET_SPECS,
                  device=DEVICE).to(DEVICE)
model.train()
train(model, OUTPUT_ROOT, mcbox_spec, train_dataloader, test_dataloader,
            device=DEVICE, epochs=10000,
            render_fn=render_cartpole,
            lr=1e-3)

