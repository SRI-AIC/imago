import argparse
import yaml

from tqdm import tqdm
from pathlib import Path
from collections import namedtuple
from matplotlib import pyplot as plt
import io
from PIL import Image

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from imago.domains import forward
from imago.perturb import Perturb
from imago.domains.reaverassaulty2.ray2_setup import *
from imago.utils import tile_images_vert, stats, scan_stats, ensure_numpy
from imago.io_utils import load_object, save_object
from imago.analysis.walk import render_traj


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
N_NEIGHBORS = config['neighbors']
results_dir = Path(config['results_dir'], "neighbors")

print(str(config))

# Collect episode and frame schedules
schedule = config['scenes']
perturb_var = config['varname']
perturb_dir = config['direction']
D = Perturb(perturb_var, direction=perturb_dir, scale=0.1)

model = load_model(device=DEVICE)
model.eval()

train_ds, test_ds = load_data(load_debug=DEBUG)

def get_snapshot(ep, frame):
    if train_ds.contains_epfr(ep, frame):
        datum = train_ds.get(ep, frame)
    elif test_ds.contains_epfr(ep, frame):
        datum = test_ds.get(ep, frame)
    else:
        raise Exception("Ep={} Frame={} not in dataset!".format(ep, frame))
    O = datum[OBS]
    return render_ray2(O, That=datum), O, datum


def get_meta(datum):
    Ohat, z_mu, z_logvar, Z, That = forward(model, datum)
    return Ohat, Z, That


IDX = "idx"
DATUM = "datum"
ODIFF = "ODiff"
EP = "ep"
FRAME = "frame"

def get_NN(O):
    nn_tuples = []
    for idx in tqdm(range(0, len(train_ds), 5)):
        datum = train_ds[idx]
        ep, fr = train_ds.get_epfr(idx)
        Op = datum[OBS]
        odiff = compute_Odiff(np.expand_dims(O, 0), np.expand_dims(Op, 0))
        nn_tuples.append({IDX: idx, DATUM: datum, ODIFF: odiff, EP: ep,
                          FRAME: frame})
    return sorted(nn_tuples, key=lambda x: x[ODIFF])



#
# Get the most confident scene, identify the neighborhood around it.
# Get statistics of each of the Interestingness Variables
#

def process(ep, frame, name,
            perturb,
            results_dir="results",
            N=20):
    probe_img, probe_O, probe_datum = get_snapshot(ep, frame)
    probe_Ohat, probe_Z, probe_That = get_meta(probe_datum)
    probe_Z = ensure_numpy(probe_Z)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    probe_img.save(Path(results_dir, "query.png"))

    probe_neighbors = get_NN(probe_O)

    # Stats on diff
    odiffs = []
    for neighbor_tuple in probe_neighbors[0:N]:
        odiff = neighbor_tuple[ODIFF]
        odiffs.append(odiff)
    print("Odiff", stats(odiffs))

    # Stats on vars
    all_vars = {}
    for neighbor_tuple in probe_neighbors[0:N]:
        datum = neighbor_tuple[DATUM]
        for k, v in datum.items():
            if k != OBS:
                if k not in all_vars:
                    all_vars[k] = []
                all_vars[k].append(v)
    for k, v in all_vars.items():
        print(k, stats(v))

    # Train up a classifier on high and low end of
    # the target variables
    tgt_var = perturb.var_name
    selected_neighborhood = probe_neighbors[0:N]
    var_mu, var_std, var_xmin, var_xmax = scan_stats(all_vars[tgt_var])
    high_insts = []
    low_insts = []
    middle_insts = []

    Zs = []
    Y = []
    # For now, POS label is negative (go low)
    for neighbor_tuple in selected_neighborhood:
        datum = neighbor_tuple[DATUM]
        Oh, Zh, Th = get_meta(datum)
        Zs.append(Zh)
        var_value = datum[tgt_var]
        if var_value >= (var_mu + var_std / 2):
            high_insts.append(neighbor_tuple)
            if perturb.direction > 0:
                Y.append(1)
            else:
                Y.append(0)
        elif var_value <= (var_mu - var_std / 2):
            low_insts.append(neighbor_tuple)
            if perturb.direction < 0:
                Y.append(1)
            else:
                Y.append(0)
        else:
            middle_insts.append(neighbor_tuple)
            Y.append(0)

    print(len(high_insts), len(middle_insts), len(low_insts))
    print(np.sum(Y) / len(Y))
    # TODO: At  extremes, sort middle and pull from top and bottom as needed

    # Make classification
    Y = np.array(Y)
    Zs = np.array([ensure_numpy(z).reshape((-1)) for z in Zs])
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=1337, tol=1e-5))
    scores = cross_val_score(clf, Zs, Y, cv=3, n_jobs=1)
    print(scores)

    # Set the full version
    clf_full = make_pipeline(StandardScaler(), LinearSVC(random_state=1337, tol=1e-5))
    clf_full.fit(Zs, Y)

    # Get the direction towards the positive label (negative confidence)
    dir_Z = clf_full.get_params()['linearsvc'].coef_
    end_Z = probe_Z + 100 * dir_Z

    render_traj(probe_Z, end_Z,
                Path(results_dir, "svc_dir_{}_N{}.mp4".format(name, N)),
                model,
                render_ray2,
                compute_odiff_fn=compute_Odiff)


for ep, frame in schedule.items():
    name = "ep{}_fr{}".format(ep, frame)
    print(name)
    process(ep, frame, name, D, N=N_NEIGHBORS, results_dir=results_dir)
