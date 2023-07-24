from absl import logging
logging.set_verbosity(logging.INFO)

import os
import hickle as hkl
from sklearn.preprocessing import StandardScaler

from imago.spaces.channels import ChannelSpec
from imago import PVAE_ROOT
from imago.models.model import PVaeModel
from imago.spaces.mc_box import *
from gym.spaces.box import Box

from imago.utils import *
from imago.utils import sigmoid

from torch.utils.data import TensorDataset, DataLoader

DEVICE="cuda:0"

CAML_ROOT = os.path.join(PVAE_ROOT, "..", "..")
DATA_FILE = os.path.join(CAML_ROOT, "datasets", "cameleon", "imago_rollouts_1009.hkl")
OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/pvae/demos/canniballs")
os.makedirs(OUTPUT_ROOT,exist_ok=True)

obs_space = Box(0, 2048, shape=(12, 12, 4), dtype=np.uint8)

CHANNEL_SPECS = [
    ChannelSpec("agent_pos", SPACE_TYPE.CATEGORICAL, low=0, high=1),
    ChannelSpec("opponents", SPACE_TYPE.CATEGORICAL, low=0, high=1),    
    ChannelSpec("food", SPACE_TYPE.CATEGORICAL, low=0, high=1),
    ChannelSpec("obstacles", SPACE_TYPE.CATEGORICAL, low=0, high=1)    
    ]

LATENT2TARGET_SPECS = [
    ChannelSpec("action_dist", SPACE_TYPE.DIST_JSD, shape=(4,)),
    ChannelSpec("action_logits", SPACE_TYPE.DIST_JSD, shape=(4,), low=0, high=1),
    ChannelSpec("value_function", SPACE_TYPE.NUMERIC, shape=(1,), low=0, high=1, apply_whitening=True)
]

mcbox_spec = MCBoxSpec(CHANNEL_SPECS, obs_space, channel_axis=-1)

model = PVaeModel(mcbox_spec, latent2target_specs=LATENT2TARGET_SPECS,
                  device=DEVICE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())

print(model)

# Load the data, and use just the obsevations for now.
D = hkl.load(DATA_FILE)

O = D['observation']
A = D['action_dist']

# Apply transforms as needed to convert numerics into targets
L = D['action_logits']
L = sigmoid(L)
V = D['value_function']
V = V.reshape((V.shape[0], 1))

BATCH_SIZE=64
EPOCHS = 10000

val_idx = -1000
train_tensors = []
test_tensors = []

# Standardize
v_scaler = StandardScaler()
v_scaler.fit(V[0:val_idx])
V = v_scaler.transform(V)

for X in [O, A, L, V]:
    train_tensors.append(torch.Tensor(X[0:val_idx]))
    test_tensors.append(torch.Tensor(X[val_idx:]))


train_dataloader = DataLoader(TensorDataset(*train_tensors), batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(TensorDataset(*test_tensors), batch_size=BATCH_SIZE, shuffle=True)

for datum in test_dataloader:
    Ohat, Z_muhat, Z_logvarhat, Zhat, That = model(datum[0])
    break

print("GOLD")
print(datum[1:])

print("\n\nNO TRAIN")
print(That)

# Now load the model
load_checkpoint(os.path.join(OUTPUT_ROOT, "model.pt"), model, None)

Ohat2, Z_muha2t, Z_logvarhat2, Zhat2, That2 = model(datum[0])

print("\n\nTRAINED")
print(That2)
