import pdb

from tqdm import tqdm
import scipy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from imago.analysis.anom.twostep_recon import TwoStepODiffAnomDet
from imago.utils import ensure_torch

def to_var(x, requires_grad=True, volatile=False, device='cuda:0'):
    x = ensure_torch(device, x)
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def realism_loss(Z, domain, update_norm_scale=0.1):
    """ Convenience routine for bumping a latent in the direction of more ``realism.''"""
    Z2 = to_var(Z)
    Ohat22, That22 = domain.model.forward_Z(Z2, ret_for_training=False)
    Ohat3, _, _, _, That3 = domain.model(Ohat22, ret_for_training=True)
    loss_rec = domain.model.recon_loss(Ohat3, Ohat22.detach())
    loss_rec.backward()
    if Z2.grad is not None:
        update_rec = Z2.grad / torch.linalg.norm(Z2.grad).item() * update_norm_scale
    else:
        update_rec = 0.
    return update_rec


import pdb

def realism_loss_gs(Z, domain):
    """ Convenience routine for bumping a latent in the direction of more ``realism.''
    Uses golden section search in the direction mandated by the gradient update"""
    Z2 = to_var(Z)
    Ohat22, That22 = domain.model.forward_Z(Z2, ret_for_training=False)
    Ohat3, _, _, _, That3 = domain.model(Ohat22, ret_for_training=True)
    loss_rec = domain.model.recon_loss(Ohat3, Ohat22.detach())
    loss_rec.backward()
    if Z2.grad is not None:
        update_dir = Z2.grad
        update_dir /= torch.linalg.norm(update_dir)
        # Now do a golden section search along this point, minimizing anomaly score
        def _inner(alpha):
            Z3 = Z2 + alpha * update_dir
            anom_model = TwoStepODiffAnomDet(domain.model, domain.odiff_fn)
            score, _ = anom_model.score(Z3)
            return score
        computed_alphas = []
        for alpha in range(0, 11, 1):
            alpha = alpha / 10.
            score = _inner(alpha)
            computed_alphas.append((alpha, score))
        idx = np.argmin([x[1] for x in computed_alphas])
        computed_alpha = computed_alphas[idx][0]
        #computed_alpha = scipy.optimize.golden(_inner, brack=(0,1), tol=1e-2, maxiter=10)
        return update_dir * computed_alpha
    else:
        return 0.
    return update_rec


class GradPerturber:
    def __init__(self, domain, perturb):
        self.domain = domain
        self.perturb = perturb

    def process(self, Z, max_iterations=100,
                outcome_norm_scale=5,
                update_norm_scale=1,
                use_anom_loss=True,
                verbose=True):
        """
        Performs gradient based updates iteratively, up until max_iterations, returning
        the resulting reconstructions and Z.
        """
        prev_value = None
        start_value = None
        target_value = None
        iter_obj = enumerate(range(max_iterations))
        if verbose:
            iter_obj = tqdm(iter_obj, total=max_iterations)
        for idx, _ in iter_obj:
            Z = to_var(Z)  # TODO: Ensure we are not flooding the graph with so many Variables
            Ohat2, That2 = self.domain.model.forward_Z(Z, ret_for_training=False)
            curr_value = That2[self.perturb.var_name]
            if prev_value is None:
                start_value = curr_value.item()
                target_value = Variable(ensure_torch(self.domain.model.device,
                                                     np.array(self.perturb.get_target_value(start_value))))
                #target_value = Variable(curr_value) + self.perturb.direction * self.perturb.tgt_mag
            loss = F.mse_loss(curr_value, target_value)
            loss.backward()
            grad_norm = torch.linalg.norm(Z.grad).item()
            update_sigma = (Z.grad / grad_norm *  outcome_norm_scale).detach()
            if use_anom_loss:
                # Zero the Z grad, so we can compute update for likelihood recon loss
                update_rec = realism_loss(Z, self.domain)
            else:
                # No anom loss applied
                update_rec = 0.

            # print(idx, curr_value)
            if verbose:
                iter_obj.set_description("Start/Curr value={:.5f}/{:.5f}".format(start_value, curr_value.item()))
            if self.perturb.target_met(start_value, curr_value.item()):
                break
            else:
                Z = Z - update_sigma - update_rec
            prev_value = curr_value.item()
        return Ohat2, That2, Z, idx