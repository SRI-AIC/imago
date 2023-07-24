import torch
import numpy as np

from scipy.spatial.distance import jensenshannon
from sc2recorder.utils import ensure_numpy

from tqdm import tqdm

def compute_divergence(model, test_dataloader, device="cpu"):
    print("Computing divergence")
    sum_divg = 0.
    Ds = []
    for X, Y, C in tqdm(test_dataloader):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        Y = torch.tensor(Y, dtype=torch.long).to(device)
        C = torch.tensor(C, dtype=torch.float32).to(device)
        D = ensure_numpy(evaluate_model(model, X, C=C, device=device))
        sum_divg += np.sum(D)
        Ds.extend(D.tolist())
    std_divg = np.std(Ds)
    mean_divg = np.mean(Ds)
    print("Divergence mean={:.5f}, std={:.5f}".format(mean_divg, std_divg))
    return Ds, mean_divg, std_divg


def pixarray_jsd(Y_hat1, Y_hat2):
    """ Flattens the semantic pixarray hypotheses, treating the exploded view as a probability distribution.
    These distributions are then measured by earth movers distance.
    Expects Y_hats to be shaped (batch_size, units, height, width)

    NOTE: Unless specified, unit 0 (empty unit) is removed, so we focus mass primarily on
    the seen units.
    """
    if len(Y_hat1.shape) != 4 or len(Y_hat2.shape) != 4:
        raise Exception("Y_hat1 and Y_hat2 must have shape of form (batch, unit, height, width")
    # Do the recommended detach and clone, in order to generate modified copies.
    Y_hat1 = Y_hat1.clone().detach()
    Y_hat2 = Y_hat2.clone().detach()
    Y_hat1[: ,0 ,: ,:] = 0
    Y_hat2[: ,0 ,: ,:] = 0
    fy1 = ensure_numpy(flatten_semantic_pixarray(Y_hat1)) + 1e-5
    fy2 = ensure_numpy(flatten_semantic_pixarray(Y_hat2)) + 1e-5
    # W = np.array([entropy(fy1[idx, :], fy2[idx, :]) for idx in range(fy1.shape[0])])
    W = np.array([jensenshannon(fy1[idx, :], fy2[idx, :]) for idx in range(fy1.shape[0])])
    return W


def evaluate_model(model, X, C=None, device="cpu"):
    """ Convenience routine for computing distance on a given model.  If C is
    given, passes it through as part of the forward call.

    Attempts to see how well reconstruction occurs against the prediction.
    X = input to model  (batch, inputs)
    C = conditioning (batch, c_dim)
    """
    if model.is_conditional():
        res = model.forward(X, C)
        if isinstance(res, dict):
            Y_hat = res['x']
        else:
            Y_hat, Mu, Logvar, Z = res
    else:
        Y_hat, Mu, Logvar, Z = model.forward(X)
    batch_size, n_units, height, width = X.shape
    Y_hat = Y_hat.reshape(batch_size, n_units, height, width)

    # Convert the integer-type targets into an onehot semantic pixel array
    # Note: This uses scatter, which is still a bit confusing for me.  However, I did vette
    # this against the manual assignment of softmaxes.  Apparently scatter takes an input value(s)
    # and assigns it to the desired axis and indices.  Here, scatter  assigns the value 1 to positions
    # indexed by max_idx on dimension 1.
    max_idx = torch.argmax(Y_hat, 1, keepdim=True)
    Y_hat_onehot = torch.zeros((batch_size, n_units, height, width)).to(device).scatter_(1, max_idx, 1)

    divergences = pixarray_jsd(Y_hat_onehot, X)
    return divergences


def reform_semantic_pixarray(y_hat, n_targets, width, height):
    """ Given a flat batch x inst shaped y_hat, converts this into
    a shape (batch, n_targets, width, height) semantic pixel array suitable
    for  multinomial_loss_fn.
    """
    return y_hat.reshape((-1, n_targets, width, height))


def flatten_semantic_pixarray(y_hat):
    batch_size = y_hat.shape[0]
    return y_hat.reshape((batch_size, -1))
