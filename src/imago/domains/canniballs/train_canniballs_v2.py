"""
Usage:
cd caml
python -m pvae.datasets.canniballs.train_canniballs_v2
"""

import sys

from torch.utils.data import DataLoader
from imago.domains.canniballs.canniballs_v2 import *

from imago import CAML_ROOT
from imago.models.model import PVaeModel
from imago.trainer import train

from absl import logging
logging.get_absl_handler().python_handler.stream = sys.stdout

DEVICE = "cuda:0"
OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/pvae/canniballs_v2_lr1e-4_W10_adamW_categorical_confANDgoalANDriskANDincongANDvalue_DEBUG")

BATCH_SIZE=1500
train_dataset, test_dataset = load_data()
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = PVaeModel(mcbox_spec, latent2target_specs=LATENT2TARGET_SPECS,
                  device=DEVICE).to(DEVICE)
model.train()
train(model, OUTPUT_ROOT, mcbox_spec, train_dataloader, test_dataloader,
            device=DEVICE, epochs=10000,
            lr=1e-4)

