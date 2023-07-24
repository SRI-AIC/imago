from torch.utils.data import DataLoader

from imago.domains.ray2_recon_only.ray2_ro_setup import *
from imago.trainer import train

DEVICE="cuda:0"
OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/pvae/RAY2_RO_lr1en3_Wrecon10")
BATCH_SIZE=16

print("Loading data")

train_ds, test_ds = load_data(load_debug=False)
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

print("Loading model")

model = load_model(device=DEVICE, force_new=True)
model.train()
train(model, OUTPUT_ROOT, mcbox_spec, train_dataloader, test_dataloader,
      device=DEVICE, epochs=10000, lr=1e-3, W_recon=10,
      step_limit=16395696)  # Added in step limit for replicability