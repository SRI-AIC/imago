"""
Load the PySC2 Y2 model, perturb and get results

Usage

From Imago root,

    python -m imago.domains.reaverassaulty2.multivar_opt_cf imago/domains/reaverassaulty2/config/config.yaml

See imago/domains/reaverassaulty2/config/config.yaml for an example of how the scenes are configured.

"""
import sys
import argparse
import yaml
from pathlib import Path

from imago.domains import OBS
from imago.domains.reaverassaulty2.ray2_setup import *
from imago.perturb import Perturb
from imago.perturb.sampler import name2fn
from imago.perturb.analysis import scan

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to config listing episode and frame offsets to perturb")
args = parser.parse_args()
print(args)

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

print(str(config))

DEVICE = config['device']
DEBUG = config['debug']
results_dir = Path(config['results_dir'])

train_ds, test_ds = load_data(load_debug=DEBUG)

# Collect episode and frame schedules
schedule = config['scenes']
perturb_var = config['varname']
perturb_dir = config['direction']
D = Perturb(perturb_var, direction=perturb_dir, scale=0.1)
method = config['method']
# "nm"      # nm | nm_strict | sample | cma
sampler_fn = name2fn(method)

model = load_model(device=DEVICE)
model.eval()

if D.direction > 0:
    dir_prefix = "inc"
else:
    dir_prefix = "dec"
shortname = var_shortname[D.var_name]
prefix = "{}_{}".format(dir_prefix, shortname)

NUM_SEARCHES=1
NUM_SAMPLES = 100
rng = np.random.default_rng(117)

for ep, fr_idx in schedule.items():
    start_datum = train_ds.get(ep, fr_idx)
    O = start_datum[OBS]
    O = ensure_torch(DEVICE, O)
    O = O.unsqueeze(0)

    print("Analyzing scene, ep={} fr={}".format(ep, fr_idx))
    res_dir = Path(
        results_dir,
        "{}_{}/ep={}_fr={}".format(prefix, method, ep, fr_idx)
    )
    scan(O, model, D, render_fn=render_ray2, num_samples=NUM_SEARCHES,
         sampler_fn=sampler_fn,
         summary_dir=res_dir)