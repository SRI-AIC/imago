"""
Load the Canniballs v2 model, perturb and get results

Usage

From Imago root,

    python -m pvae.datasets.canniballs.perturb_analysis

"""
import numpy as np

from imago.domains import OBS
from imago.domains.cartpole.cartpole_setup import *
from imago.perturb.sampler import *
from imago.perturb.analysis import scan
DEVICE="cuda:0"

train_ds, test_ds = load_data(add_spatial_dims=True)
model = load_model(device=DEVICE)
model.eval()


def filter_by_perturb(perturb, factor=1.):
    """
    Given a perturbation, identify the starting points that
    are past 1 std in the opposite of the desired direction
    of perturbation.

    Returns the offsets in the original train_ds corresponding to
    good starting points
    :param perturb:
    :return:
    """
    varname = perturb.var_name
    V = np.array([datum[varname].flatten()[0] for datum in train_ds])
    mu, std = np.mean(V), np.std(V)
    if perturb.direction > 0:
        idxes = np.where(V < (mu - std) / factor)[0]
    elif perturb.direction < 0:
        idxes = np.where( V > (mu + std) / factor)[0]
    else:
        raise Exception("Equality perturbation not supported!")
    assert len(idxes) > 0
    return idxes


#D = Perturb(CONFIDENCE, direction=-1, scale=1)
# D = Perturb(RISKINESS, direction=-1, scale=1)
D = Perturb(RISKINESS, direction=1, scale=1)
prefix = "inc_risk"

method = "nm"  # nm | nm_strict | sample

if method == "nm":
    sampler_fn = nm_search_simp
elif method == "nm_strict":
    sampler_fn = nm_search
elif method == "sample":
    sampler_fn = psearch
elif method == "cma":
    sampler_fn = cma_search
elif method == "bfgs":
    sampler_fn = bfgs_search
else:
    raise Exception("Unsupported sampler function={}".format(method))

NUM_SAMPLES = 100
rng = np.random.default_rng(117)

scene_idxes = filter_by_perturb(D, factor=1.5)
#scene_idxes = np.sort(rng.choice(np.arange(0, len(train_ds)), NUM_SAMPLES))


print("Valid scenes={}".format(scene_idxes))

for scene_idx in scene_idxes:
    start_datum = train_ds[scene_idx]
    O = start_datum[OBS]
    O = ensure_torch(DEVICE, O)
    O = O.unsqueeze(0)

    scan(O, model, D, render_fn=render_cartpole, num_samples=8,
         sampler_fn=sampler_fn,
         summary_dir="results/cartpole/{}_{}/scene_{}".format(prefix, method, scene_idx),
         #summary_dir="results/cartpole/dec_conf_{}_v2/scene_{}".format(method, scene_idx),
         min_Odiff=1e-1)