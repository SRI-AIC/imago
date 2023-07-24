import numpy as np
from tqdm import tqdm
from pathlib import Path

from imago.domains import OBS, summary_str
from imago.domains.reaverassaulty2.ray2_setup import *
from imago.utils import ensure_torch

DEBUG=True
DEVICE="cuda:0"
if DEBUG:
    OUTPUT_DIR = Path(DATA_ROOT, "start_scenes_DEBUG")
else:
    OUTPUT_DIR = Path(DATA_ROOT, "start_scenes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Output dir={}".format(OUTPUT_DIR.absolute()))

train_ds, test_ds = load_data(load_debug=DEBUG)

"""
TODO:
- See distances for start scenes in Z space against all other scenes
- See perturability
"""

def scan_startscenes(prefix, dataset):
    start_scenes = []
    D1 = dataset[0]
    O1 = ensure_torch(DEVICE, np.expand_dims(D1[OBS], 0))
    imgs_dir = Path(OUTPUT_DIR, "images_{}".format(prefix))
    imgs_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(OUTPUT_DIR, "{}_start_idxes.csv".format(prefix)), 'w') as f:
        def add_scene(datum, idx):
            start_scenes.append(idx)
            img = render_ray2(datum)
            img.save(Path(imgs_dir, "{}_start_scene_{}.png".format(prefix, idx)))
            f.write("{}\n".format(str(idx)))
        add_scene(D1, 0)
        for idx in tqdm(range(1, len(dataset))):
            datum = dataset[idx]
            O2 = ensure_torch(DEVICE, np.expand_dims(datum[OBS], 0))
            diff = compute_Odiff(O1, O2)
            if diff > 100:
                add_scene(datum, idx)
            O1 = O2

scan_startscenes("train", train_ds)
scan_startscenes("test", test_ds)