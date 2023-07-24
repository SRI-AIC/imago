import sys, os
from torch.utils.data import DataLoader

from imago.domains.pysc2.pysc2 import *

from imago import CAML_ROOT
from imago.trainer import train

from absl import logging
logging.get_absl_handler().python_handler.stream = sys.stdout

DEVICE = "cuda:0"
OUTPUT_ROOT = os.path.join(CAML_ROOT, "output/pysc2/caml_y1")
BATCH_SIZE = 24

DEBUG=False

train_dataset, test_dataset, v_mean, v_std = load_y1_data(debug=DEBUG)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = setup_y1_model(device=DEVICE)
model.train()
if True:
    train(model, OUTPUT_ROOT, mcbox_spec, train_dataloader, test_dataloader,        
            device=DEVICE, epochs=10000,
            lr=1e-5)



