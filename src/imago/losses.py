import collections
import torch
from torch.nn import functional as F


RECON = "recon_loss"
KL = "KL_loss"
VALUE = "value_loss"


# Behavioral distribution matching measures
MSE, JSD, KLDIV = "mse", "jsd", "kldiv"

# Evaluate the accuracy
def argmax_check(y_hat, y):
    """
    Given two torch arrays where the softmax is dim=1, counts up 
    matches and returns count."""
    y_hat = torch.argmax(y_hat, dim=1)
    y = torch.argmax(y, dim = 1)
    return torch.sum(y == y_hat).item()
    

def jsd(x, y, lm=1e-10):
    m = (x + y + 2*lm) / 2
    ret = 0.5 * (F.kl_div(torch.log(x + lm), m) + F.kl_div(torch.log(y + lm), m))
    return ret
    
    
def dist_divg_fn(y_hat, y, selected_fn):
    if selected_fn == JSD:
        return jsd(y_hat, y)
    elif selected_fn == MSE:
        return F.mse_loss(y_hat, y)
    elif selected_fn == KLDIV:
        return F.kl_div(torch.log(y_hat), y)
    raise Exception("Unknown logit loss function={}".format(selected_fn))
    
    
def model_loss(vae_model, Os, Ts={}, W=1,
               beta=1, named_losses=None,
               outcome_only=False):
    """
    TODO: Repair losses to match in original data space
    """
    if named_losses is None:
        named_losses = collections.OrderedDict()


    if not(outcome_only):
        Yhat, Z_mu, Z_logvar, Z, That = vae_model(Os, deterministic=False,
                                                  ret_for_training=True)
        loss  = W * vae_model.recon_loss(Yhat, Os)
        named_losses[RECON] = named_losses.get(RECON, 0) + loss.item()
        kl_loss = beta * torch.mean(0.5 * torch.sum(torch.exp(Z_logvar) + Z_mu**2 - 1. - Z_logvar, 1))
        loss += kl_loss
    else:
        # In case we only want to compute the outcome loss
        _, _, _, _, That = vae_model(Os, deterministic=False,
                          ret_for_training=True)
        loss = None

    # Get latent2 target heads
    T_losses, T_losses_dict = vae_model.latent2target_loss(That, Ts)
    if loss is None:
        loss = T_losses
    else:
        loss += T_losses

    if not(outcome_only):
        # KL loss only computed on reconstruction
        named_losses[KL] = named_losses.get(KL, 0) + kl_loss.item()
    named_losses.update(T_losses_dict)

    return loss, named_losses
