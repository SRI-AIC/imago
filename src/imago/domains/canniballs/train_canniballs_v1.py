import sys


from imago.domains.canniballs.canniballs_v1 import *

from imago import CAML_ROOT
from imago.models.model import PVaeModel
from imago.trainer import train

from absl import logging
logging.get_absl_handler().python_handler.stream = sys.stdout

DEVICE = "cuda:0"

DATA_FILE = os.path.join(CAML_ROOT, "datasets", "cameleon", "imago_rollouts-2578_rs42_w14.hkl")

# OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/pvae/demos/canniballs_2578_rs42_w14_demo")
#OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/pvae/canniballs_2578_rs42_w14")
#OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/pvae/canniballs_2578_rs42_w14_smaller_lr")

# See if we still get good recon
OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/pvae/canniballs_2578_rs42_w14_smaller_lr_sanity")


train_dataloader, test_dataloader = load_canniballs_data(DATA_FILE, batch_size=1500)

model = PVaeModel(mcbox_spec, latent2target_specs=LATENT2TARGET_SPECS,
                  device=DEVICE).to(DEVICE)
model.train()
train(model, OUTPUT_ROOT, mcbox_spec, train_dataloader, test_dataloader,        
            device=DEVICE, epochs=10000,
            lr=1e-4)

