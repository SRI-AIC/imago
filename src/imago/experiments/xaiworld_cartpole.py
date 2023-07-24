"""
Sets up the visualizations for the XAIWorldConference examples review
"""
import sys
import yaml
from tqdm import tqdm
import torch
from torch.autograd import Variable
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass
import collections
from pathlib import Path
import pdb

from imago import IMAGO_ROOT
from imago.utils import *
from imago.domains import forward, bformat_obs, ANOM
from imago.domains.cartpole.cartpole_setup import *
from imago.perturb import Perturb
from imago.perturb.cfs.run_experiments import CFExps, res_group2dict
USE_MP = True

RES_DIR = Path("results", "cartpole")
RES_DIR.mkdir(exist_ok=True, parents=True)

DEBUG=False
DEVICE="cuda"

USE_GRAD = True
USE_NUN = True

if DEBUG:
    MAX_GRAD_ITERATIONS = 10
else:
    MAX_GRAD_ITERATIONS = 1000

train_ds, test_ds = load_data()

game_domain = load_domain(device=DEVICE)
game_domain.model.eval()

print(len(train_ds), len(test_ds))


game_domain.model.force_deterministic=True
SEARCH_LIMIT=50  # Number of instances to scan

experiments = collections.OrderedDict()

name2varmag = {
    "value": (VALUE_FUNCTION, 2 * 7.61),  # Use 2 standard deviations
    "confidence": (CONFIDENCE, 0.5),
    "riskiness": (RISKINESS, 0.5),
    # "incongruity": (INCONGRUITY, 1e-1)
}


# Set up schedule of experiments
for colname, var_tuple in name2varmag.items():
    varname, tgt_mag = name2varmag[colname]
    for direction in [-1, 1]:
        perturb = Perturb(varname, direction, tgt_mag=tgt_mag)
        cf_exp = CFExps(game_domain, perturb, train_ds)
        expname = "{}_d{}".format(colname, direction)
        experiments[expname] = {
            "exp": cf_exp,
            "col": colname,
            "schedule": perturb.get_good_queries(train_ds, limit=SEARCH_LIMIT, search_magnitude=100)
        }

results_dfs = collections.OrderedDict()
import traceback
for exp_name, exp_tuple in experiments.items():
    cf_exp, schedule, colname = exp_tuple['exp'], exp_tuple['schedule'], exp_tuple['col'].capitalize()
    exp_results = []
    print("Experiment={}, size={}".format(exp_name, len(schedule)))
    for inst_idx in tqdm(schedule):
        try:
            exp_result = cf_exp.score_inst(inst_idx, use_grad=USE_GRAD, use_nun=USE_NUN, 
                                        verbose=True, max_grad_iterations=MAX_GRAD_ITERATIONS,
                                        assemble_cf_only=True)
            exp_results.append(exp_result)
            for method_name, method_result in exp_result.items():
                fname = f"{exp_name}.i{inst_idx}.{method_name}.mp4"
                local_resdir = Path(RES_DIR, f"{method_name}", f"{exp_name}", f"cf_met={method_result.cf_met}", f"{inst_idx}")
                local_resdir.mkdir(exist_ok=True, parents=True)
                render_fpath = Path(local_resdir, fname)
                if isinstance(method_result.start_value, torch.Tensor):
                    tgt_var_value = method_result.start_value.item()
                else:
                    tgt_var_value = float(method_result.start_value)
                method_result.render(render_fpath, render_vars=[cf_exp.perturb.var_name, ANOM], target_pairs=[(cf_exp.perturb.var_name, tgt_var_value - cf_exp.perturb.tgt_mag)])
        except Exception as ex:
            print(f"Exception processing inst={inst_idx}")
            print("".join(traceback.TracebackException.from_exception(ex).format()) == traceback.format_exc() == "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))) 
            print("".join(traceback.TracebackException.from_exception(ex).format()))

print("Run complete")
sys.exit(0)