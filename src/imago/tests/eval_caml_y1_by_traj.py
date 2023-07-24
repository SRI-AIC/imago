import os,sys

sys.path.append(os.path.join("..",".."))

from absl import logging

logging.set_verbosity(logging.INFO)

from imago import PVAE_ROOT
from imago.models.model import PVaeModel
import os

from torch.utils.data import DataLoader

from imago.utils import *
from imago.trainer import eval

from imago.domains.pysc2 import mcbox_spec, latent2_target_specs, load_y1_as_trajs

""" Demonstrates training using SC2 data structures.
Separates data at different levels by trajectories.


SPLITAT 0.5
----------
TRAIN

	Loss:	mu/std=0.00062449/0.00009194, min/max=0.00041963/0.00122481
	recon_loss:	mu/std=0.00008349/0.00003162, min/max=0.00001297/0.00021194
	KL_loss:	mu/std=0.00046129/0.00003879, min/max=0.00035963/0.00060056
	Value:	mu/std=0.00000915/0.00001179, min/max=0.00000149/0.00020282
	B0:	mu/std=0.00001038/0.00002473, min/max=0.00000000/0.00026029
	B1:	mu/std=0.00000228/0.00000803, min/max=0.00000000/0.00018750
	B2:	mu/std=0.00000524/0.00002157, min/max=0.00000001/0.00045803
	B3:	mu/std=0.00000688/0.00001431, min/max=0.00000001/0.00018587
	B4:	mu/std=0.00000594/0.00000415, min/max=0.00000111/0.00008429
	B5:	mu/std=0.00000388/0.00000668, min/max=0.00000008/0.00016649
	B6:	mu/std=0.00000698/0.00000783, min/max=0.00000057/0.00020498
	B7:	mu/std=0.00000547/0.00000352, min/max=0.00000101/0.00004626
	B8:	mu/std=0.00000089/0.00000119, min/max=0.00000001/0.00001123
	B9:	mu/std=0.00001084/0.00000456, min/max=0.00000185/0.00003851
	B10:	mu/std=0.00000974/0.00000457, min/max=0.00000181/0.00005506
	B11:	mu/std=0.00000205/0.00000187, min/max=0.00000001/0.00001470
Behavior macro average=0.00000588
----------
VAL

	Loss:	mu/std=0.04444185/0.00557537, min/max=0.02939669/0.06298172
	recon_loss:	mu/std=0.02046368/0.00149037, min/max=0.01543890/0.02628166
	KL_loss:	mu/std=0.00055399/0.00003133, min/max=0.00045464/0.00065868
	Value:	mu/std=0.00061069/0.00021431, min/max=0.00016176/0.00210291
	B0:	mu/std=0.00177941/0.00121644, min/max=0.00000008/0.00732907
	B1:	mu/std=0.00427631/0.00241333, min/max=0.00002364/0.01378752
	B2:	mu/std=0.00421357/0.00204776, min/max=0.00004497/0.01156990
	B3:	mu/std=0.00583468/0.00277701, min/max=0.00011718/0.01849850
	B4:	mu/std=0.00048749/0.00021908, min/max=0.00013558/0.00135927
	B5:	mu/std=0.00167934/0.00049764, min/max=0.00039333/0.00354626
	B6:	mu/std=0.00118084/0.00034984, min/max=0.00033248/0.00268566
	B7:	mu/std=0.00111751/0.00025453, min/max=0.00046969/0.00202821
	B8:	mu/std=0.00005846/0.00011597, min/max=0.00000004/0.00087337
	B9:	mu/std=0.00097335/0.00026394, min/max=0.00033886/0.00201615
	B10:	mu/std=0.00091871/0.00021382, min/max=0.00040225/0.00178631
	B11:	mu/std=0.00029381/0.00034122, min/max=0.00000007/0.00182032
Behavior macro average=0.00190112
"""

DEBUG = False
SPLIT_AT = 0.8
#SPLIT_AT = 0.5

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

eval(model, OUTPUT_ROOT, mcbox_spec=mcbox_spec,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader, device=DEVICE,
      epochs=1000)

