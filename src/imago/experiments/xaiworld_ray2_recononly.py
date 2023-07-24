#!/usr/bin/env python
# coding: utf-8

# # RAY2 Counterfactual analyses, using Perturb identified start points and the split model
# 
# Split model is one that trained on recon first, then trained outcomes on its latents.

# In[1]:


import yaml
import torch
from torch.autograd import Variable
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass
import collections
import pdb

from imago import IMAGO_ROOT
from imago.utils import *
from imago.domains.ray2_recon_only.ray2split_setup import *
from imago.perturb import Perturb
from imago.perturb.cfs.run_experiments import CFExps, res_group2dict
USE_MP = True


DEBUG=False
DEVICE="cuda:0"

if DEBUG:
    RES_DIR = Path("results/ray2split_auto/debug")
    SCHED_SIZE=10
else:
    RES_DIR = Path("results/ray2split_auto/full")
    SCHED_SIZE=100    
RES_DIR.mkdir(exist_ok=True, parents=True)

train_ds, test_ds = load_data(load_debug=DEBUG)

cp_domain = load_domain(device=DEVICE)
cp_domain.model.eval()

print(len(train_ds), len(test_ds))


experiments = collections.OrderedDict()

name2varmag = {
    "value": (VALUE_FUNCTION, 2 * 607.1),  # Use 2 standard deviations
    "confidence": (CONFIDENCE, 0.5),
    "riskiness": (RISKINESS, 0.5),
    # "incongruity": (INCONGRUITY, 1e-1)
}

# Set up schedule of experiments
for colname, var_tuple in name2varmag.items():
    varname, tgt_mag = name2varmag[colname]
    for direction in [-1, 1]:
        perturb = Perturb(varname, direction, tgt_mag=tgt_mag)
        cf_exp = CFExps(cp_domain, perturb, train_ds)
        expname = "{}_d{}".format(colname, direction)
        experiments[expname] = {
            "exp": cf_exp,
            "col": colname,
            "schedule": perturb.get_good_queries(train_ds, limit=SCHED_SIZE)
        }

# Get the std for the value function since it's different
values = [datum[VALUE_FUNCTION][0] for datum in train_ds]
stats(values)

results_dfs = collections.OrderedDict()

for exp_name, exp_tuple in experiments.items():
    cf_exp, schedule, colname = exp_tuple['exp'], exp_tuple['schedule'], exp_tuple['col'].capitalize()
    exp_results = []
    print("Experiment={}, size={}".format(exp_name, len(schedule)))
    for inst_idx in tqdm(schedule):
        exp_results.append(cf_exp.score_inst(inst_idx))

    res_df = []
    for res_group in exp_results:
        res_df.append(res_group2dict(res_group))
    res_df = pd.DataFrame(res_df)
    res_df.to_csv(Path(RES_DIR, "{}.csv".format(exp_name)), header=True, index=True)
    results_dfs[exp_name] = res_df

