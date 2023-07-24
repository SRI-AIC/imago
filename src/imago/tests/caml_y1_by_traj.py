import os,sys

sys.path.append(os.path.join("..",".."))

from absl import logging

logging.set_verbosity(logging.INFO)

from imago import PVAE_ROOT
from imago.models.model import PVaeModel
import os

from torch.utils.data import DataLoader

from imago.utils import *
from imago.trainer import train

from imago.domains.pysc2 import mcbox_spec, latent2_target_specs, load_y1_as_trajs

""" Demonstrates training using SC2 data structures.
Separates data at different levels by trajectories.
"""

DEBUG = False

SPLIT_AT = 0.5

# Trained models
# SPLIT_AT = 0.8

print(os.getcwd())

CAML_ROOT = os.path.join(PVAE_ROOT, "..", "..")
IDP_ROOT = os.path.join(CAML_ROOT, "output", "assault", "interaction_data")

DEBUG_STR = ""
if DEBUG:
    DEBUG_STR = "DEBUG"

EXP_NAME = "caml_y1_trajs_{}_splitat={}".format(DEBUG_STR, SPLIT_AT)
OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/pvae/demos/{}".format(EXP_NAME))
os.makedirs(OUTPUT_ROOT,exist_ok=True)

MODEL_FPATH = os.path.join(OUTPUT_ROOT, "sc2_model.pt")

DEVICE='cuda:0'

model = PVaeModel(mcbox_spec,
                  latent2target_specs=latent2_target_specs,
                  device=DEVICE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())



train_dataset, test_dataset, train_trajs, test_trajs = load_y1_as_trajs(debug=DEBUG, split_at=SPLIT_AT)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

train(model, OUTPUT_ROOT, mcbox_spec=mcbox_spec,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader, device=DEVICE,
      epochs=1000)

