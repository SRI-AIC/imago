"""
Instance based counterfactual analysis, originally
from imago/notebooks/220309_cf_inst.py
"""

import argparse
import yaml

from imago.perturb import Perturb
from imago.domains import ANOM, ODIFF
from imago.domains.reaverassaulty2.ray2_setup import *

from imago.perturb.cfs.instance_based import NeighborhoodCFScanner, get_valid_fn
"""
Scans the featurewise neighborhood and develops a direction of perturbation for
the latent vector to obtain a counterfactual.
"""

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to config listing episode and frame offsets to perturb")
args = parser.parse_args()
print(args)

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

DEVICE = config['device']
DEBUG = config['debug']
results_dir = Path(config['results_dir'], "cf_inst")

# Collect episode and frame schedules
schedule = config['scenes']
perturb_var = config['varname']
perturb_dir = config['direction']
tgt_mag = config['tgt_mag']
D = Perturb(perturb_var, direction=perturb_dir, scale=0.1,
            tgt_mag=tgt_mag)

# Valid instance selection policy
valid_limit = config.get('valid_limit', None)
if 'valid_sel_method' in config:
    valid_sel_method = get_valid_fn(config['valid_sel_method'])
else:
    valid_sel_method = None

ray2_domain = load_domain(device=DEVICE)
ray2_domain.model.eval()

# If on non-Linux system set use_mp=False
train_ds, test_ds = load_data(load_debug=DEBUG, use_mp=True)

nCF = NeighborhoodCFScanner(ray2_domain,
                            train_ds, D, results_dir=results_dir,
                            valid_inst_sel_fn=valid_sel_method)


def get_datum(ep, frame):
    if train_ds.contains_epfr(ep, frame):
        datum = train_ds.get(ep, frame)
    elif test_ds.contains_epfr(ep, frame):
        datum = test_ds.get(ep, frame)
    return datum


def get_name(ep,fr):
    return "{}_fr{}".format(ep, fr)


# Generate the neighborhood for the schedule
neighbors = {}
for ep, frame in schedule.items():
    name = get_name(ep, frame)
    print(name)
    query_datum = get_datum(ep, frame)
    nCF.process(query_datum, name=name,
                additional_vars=[ANOM, ODIFF])

