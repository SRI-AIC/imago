from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from numpy.random import default_rng
from scipy import optimize as soptim
import cma


from imago.perturb import Perturb, DeltaVar
from imago.utils import *
from imago.viz import *
import copy

"""
Implements a perturbation sampler.

Starts with a given scene and then perturbs using the model until the specific
criteria is met.  If the observational difference is zero, then increase the
scale of the perturbation and try again.

Implements a variant of the Nelder-Mead method.  Instead of the usual method
for sampling new points, we presume the desired solution is within the manifold
described by the best matching vectors, and select pairwise means from the
current solution set.
"""

V = "V" # Value
E = "E" # Entropy
O = "O" # Observation delta


def name2fn(method):
    if method == "nm":
        sampler_fn = nm_search_simp
    elif method == "nm_strict":
        sampler_fn = nm_search
    elif method == "nm_simp_large":
        sampler_fn = nm_search_simp_large
    elif method == "sample":
        sampler_fn = psearch
    elif method == "cma":
        sampler_fn = cma_search
    elif method == "bfgs":
        sampler_fn = bfgs_search
    else:
        raise Exception("Unsupported sampler function={}".format(method))
    return sampler_fn


def gen_delta(start_Z, model, perturb, rectify_mag=False):
    device = model.device
    start_Ohat, start_That = model.forward_Z(ensure_torch(device, start_Z))
    def fn(Z):
        Oh, Th = model.forward_Z(ensure_torch(device, Z))
        score = 0.
        for var_name, var_value in Th.items():
            var_delta = np.sum(ensure_numpy(var_value - start_That[var_name]))
            if var_name == perturb.var_name:
                # Objective we want to maximize delta, so translate to a minimization
                dir_mag = var_delta * perturb.direction  # Align with desired maximization
                if rectify_mag:
                    # Rectify so we don't keep trying to maximize perturbation direction,
                    # at the cost of adding noise to the other variables.
                    dir_mag = max(dir_mag, 1)
                score -= dir_mag # Reverse, so we minimize desired direction
            else:
                # Regular, minimize delta
                score += abs(var_delta)
        return score
    return fn


def nm_search(start_Z, model,
              perturb: Perturb, device="cpu", verbose=False):
    soln = soptim.minimize(gen_delta(start_Z, model, perturb),
                           ensure_numpy(start_Z), method="Nelder-Mead")
    return soln.x


def nm_search_simp(start_Z, model,
                   perturb: Perturb,
                   rectify=False,
                   scale=1.0,
                   device="cpu", verbose=False):
    """
    NM, but starts with a simplex sampled around start_Z
    :param start_Z:
    :param model:
    :param perturb:
    :param device:
    :param verbose:
    :return:
    """
    num_samples = start_Z.shape[-1] + 1 # N+1 samples to cover each dimension
    start_Z = ensure_numpy(start_Z)
    # W = np.random.randn(num_samples, start_Z.shape[-1])
    W = np.random.randn(num_samples, start_Z.shape[-1]) * scale
    # Rectify W
    if rectify:
        W = np.maximum(W, -1)
        W = np.minimum(W, 1)

    S = start_Z + W
    soln = soptim.minimize(gen_delta(start_Z, model, perturb),
                           start_Z, method="Nelder-Mead",
                           options= {
                               "initial_simplex": S
                           })
    return soln.x

def nm_search_simp_large(start_Z, model,
                   perturb: Perturb,
                   rectify=False,
                   device="cpu", verbose=False):
    return nm_search_simp(start_Z, model, perturb=perturb,
                          rectify=rectify, scale=100, device=device, verbose=verbose)

def bfgs_search(start_Z, cost_fn):
    soln = soptim.minimze(cost_fn, start_Z, method='BFGS',
                          options={
                              "display": True
                          })
    return soln.x


def cma_search(start_Z, model,
               perturb:Perturb, device="cpu", verbose=False):
    es = cma.CMAEvolutionStrategy(ensure_numpy(start_Z).flatten(), 0.1)
    soln = es.optimize(gen_delta(start_Z, model, perturb),
                       maxfun=20000)
    return soln.best.x


def psearch(start_Z,
            model,
            initial_perturb: Perturb,
            num_samples=10000,
            num_epochs=10,
            N=3,  # Top N to retain for searching
            device="cpu",
            verbose=False):
    """ Performs a gradient free linearization-style pursuit using particles"""
    rng = default_rng()
    perturb = copy.copy(initial_perturb)
    start_Z = ensure_torch(device, start_Z)
    start_Ohat, start_That = model.forward_Z(start_Z)
    start_Ohat = ensure_numpy(start_Ohat)

    # Get initial stab and get deltas by variable
    W = perturb.scale * np.random.randn(num_samples, 256) # num_samples x size
    W = ensure_torch(device, W)
    end_Zs = start_Z + W # Initial candidate set

    # Main loop for searching valid directions
    for epoch in tqdm(range(num_epochs)):
        # Create dataloader for generating probes, so we can iterate through and collect them
        dl = DataLoader(TensorDataset(end_Zs, torch.ones(len(end_Zs))),
                        batch_size=10, shuffle=False)

        # Iterate through all candidates and assemble the delta of the probe against
        # the starting point.
        step_deltas = {}
        obs_deltas = []
        for Zs, Ys in dl:
            Ohat3, That3 = model.forward_Z(Zs)
            Ohat3 = ensure_numpy(Ohat3)
            obs_deltas.append(DeltaVar(O, Ohat3 - start_Ohat))
            # For each of the targets, get the deltas
            for var_name, var_value in That3.items():
                if var_name not in step_deltas:
                    step_deltas[var_name] = []
                start_value = start_That[var_name]
                var_delta = ensure_numpy(var_value - start_value)
                if len(var_delta.shape) > 1:
                    var_delta = np.sum(var_delta, axis=-1) # Sum along start_Zs, so we can rank probe directions
                step_deltas[var_name].append(var_delta)
        for var_name, delta_v in step_deltas.items():
            step_deltas[var_name] = np.concatenate(step_deltas[var_name])
        obs_deltas = DeltaVar(O, np.concatenate([d.dv for d in obs_deltas]))

        if verbose:
            print("Epoch:{}".format(epoch))
            for varname, dvar in step_deltas.items():
                print(dvar.stats_str())
        
        # Collect early stopping criteria:
        # - variance amongst Z probes is sufficiently small
        stop_loop = False        
        delta_Z_var = torch.mean(torch.var(end_Zs, dim=0)).item()
        if delta_Z_var == 0:
            stop_loop =True
        
        # If no early stopping or not at end, then collect new simplex
        # by averaging candidates
        if stop_loop:
            break
        else:
            min_indices = {} # Probes minimizing changes in other variables
            max_indices = {} # Probes maximizing change in target variable
            for varname, dvars in step_deltas.items():
                # Minimize magnitude of delta if this variable is not
                # the one we wish to perturb
                if varname != perturb.var_name:
                    min_indices[varname] = np.argsort(np.abs(dvars))[0:N]
                else:
                    if perturb.direction is None:
                        dvars = np.abs(dvars) # Look for magnitude (regardless of direction)
                    else:
                        dvars = dvars * perturb.direction # Directional
                    max_indices[varname] = np.argsort(dvars)[-N:]
            # Get the new set of candidate particles
            new_Zs = []
            # Generate new centroids based off of combinations of min and max
            # First generate complete centroid
            if True:
                combined = [ensure_numpy(end_Zs[v]) for v in min_indices.values()] + \
                           [ensure_numpy(end_Zs[v]) for v in max_indices.values()]
                combined = np.concatenate(combined)
                combined = np.mean(combined, axis=0)
                new_Zs.append(combined)

            # Randomized centroids
            if False:
                for _ in range(N * 10):
                    rnd_combined = [rng.random() * ensure_numpy(end_Zs[v]) for v in min_indices.values()] + \
                                   [rng.random() * ensure_numpy(end_Zs[v]) for v in max_indices.values()]
                    rnd_combined = np.concatenate(rnd_combined)
                    rnd_combined = np.mean(rnd_combined, axis=0)
                    new_Zs.append(rnd_combined)

            # Take the centroids of desirable probes in the minimal direction
            # and those in the maximal direction, and average them to get a new set.
            if True:
                min_indices, max_indices  = np.concatenate(list(min_indices.values())), np.concatenate(list(max_indices.values()))
                for z1 in end_Zs[min_indices]:
                    for z2 in end_Zs[max_indices]:
                        z3 = (z1 + z2) / 2
                        new_Zs.append(z3)

            del end_Zs
            end_Zs = torch.stack([ensure_torch(device, z) for z in new_Zs])
    return end_Zs[0] # Should have converged by this point

