import os,sys

sys.path.append(os.path.join("..",".."))

from absl import logging

logging.set_verbosity(logging.INFO)

from imago import PVAE_ROOT
from imago.models.model import PVaeModel
import os
from glob import glob

from imago_prev.data.datasets import IDPDataset
from torch.utils.data import DataLoader

from imago.utils import *
from imago.trainer import train

from imago.domains.pysc2 import mcbox_spec, latent2_target_specs, load_idps

""" Demonstrates training using SC2 data structures
"""

DEBUG = False
print(os.getcwd())

CAML_ROOT = os.path.join(PVAE_ROOT, "..", "..")
IDP_ROOT = os.path.join(CAML_ROOT, "output", "assault", "interaction_data")
OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/pvae/demos/caml_y1")
os.makedirs(OUTPUT_ROOT,exist_ok=True)

MODEL_FPATH = os.path.join(OUTPUT_ROOT, "sc2_model.pt")

DEVICE='cuda:0'

model = PVaeModel(mcbox_spec,
                  latent2target_specs=latent2_target_specs,
                  device=DEVICE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())

idp_files = glob(os.path.join(IDP_ROOT, "**", "**", "interaction_data.pkl.gz"))

if DEBUG:
    idp_files = idp_files[0:10]

print("Root={}, Total IDP files={}".format(os.path.abspath(IDP_ROOT), len(idp_files)))

all_idps = load_idps(idp_files)

train_dataset = IDPDataset(all_idps[0:-100])
v_mean, v_std = train_dataset.compute_V_params()
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = IDPDataset(all_idps[-100:])
test_dataset.set_V_params(v_mean, v_std)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

train(model, OUTPUT_ROOT, mcbox_spec=mcbox_spec,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader, device=DEVICE,
      epochs=1000)

