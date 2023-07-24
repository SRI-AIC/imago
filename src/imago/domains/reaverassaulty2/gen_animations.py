from pathlib import Path
from torch.utils.data import DataLoader

from imago.domains.reaverassaulty2.ray2_setup import *
from imago.trainer import train
from imago.viz import *

DEVICE="cuda:0"
OUTPUT_ROOT = Path(CAML_ROOT, "output/animations/ReaverAssaultY2")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Loading data")

train_ds, test_ds = load_data()
START_IDX = 0
END_IDX = 3000

gif_fpath = Path(OUTPUT_ROOT,
                 "animated_{}_{}.gif".format(START_IDX, END_IDX))
print("Generating gif, writing to {}".format(gif_fpath))

render_ds2gif(train_ds, START_IDX, END_IDX, space_specifier=mcbox_spec, gif_fpath=gif_fpath)