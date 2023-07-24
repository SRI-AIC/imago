import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .reaver_behav import REAVER_ACT_SPECS
from sc2recorder.utils import ensure_numpy, ensure_torch, ensure_torch_long, set_random_seed

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
    
import pdb
def vae_loss(vae_model, Os, beta=1, named_losses=None):
    if named_losses is None:
        named_losses = {}
    y2, z_mu, z_logvar, z = vae_model.forward(Os)  
    loss  = vae_model.sc2_mf_out.loss_fn(y2, Os)
    named_losses[RECON] = named_losses.get(RECON, 0) + loss.item()
    kl_loss = beta * torch.mean(0.5 * torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1))
    loss += kl_loss
    named_losses[KL] = named_losses.get(KL, 0) + kl_loss.item()
    return loss, named_losses


def behavior_losses(behav_model, Os, Bs, V, dist_loss_type, named_losses = None, correct_by_act=None, device="cpu"):
    if named_losses is None:
        name_losses = {}
    if correct_by_act is None:
        correct_by_act = {}
    Vhat, Bhats = behav_model(Os)
    
    # Get the Value function loss
    loss = F.mse_loss(V.view(-1), Vhat.view(-1))
    named_losses[VALUE] = named_losses.get(VALUE, 0) + loss.item()
    
    # Go through each of the action probabilities
    for act_idx, act_spec in enumerate(behav_model.reaver_act_spec):
        b = Bs[act_idx]
        bhat = Bhats[act_idx]
        curr_dist_loss = dist_divg_fn(bhat, b, dist_loss_type)
        loss += curr_dist_loss
        loss_name = "dist_divg_loss_{}_{}".format(act_spec[0], dist_loss_type)
        named_losses[loss_name] = named_losses.get(loss_name, 0) + curr_dist_loss.item()
        correct_by_act[act_spec] = correct_by_act.get(act_spec, 0) + argmax_check(b, bhat)
    return loss, named_losses, correct_by_act
        

def behavior_losses_old(behav_model, Os, Bs, Vs, logit_loss_type, named_losses = None, correct_by_act=None, device="cpu"):
    if named_losses is None:
        named_losses = {}
    if correct_by_act is None:
        correct_by_act = {}
    # Behavioral and value targets
    Vhat, Bhats = behav_model(Os)
    loss = F.mse_loss(Vs.view(-1), V_hats.view(-1))
    named_losses[VALUE] = named_losses.get(VALUE, 0) + loss.item()
    for B, Bhat in zip(Bs, Bhats):
        # Go through each of the action types and collect the appropriate loss
        for act_idx in range(len(REAVER_ACT_SPECS)):
            b = B[act_idx]
            y_hat = Y_hat[act_idx]
            act_spec = REAVER_ACT_SPECS[act_idx][0]
            curr_logit_loss = dist_divg_fn(y_hat, y, logit_loss_type)
            loss += curr_logit_loss
            loss_name = "logit_loss_{}_{}".format(act_spec, logit_loss_type)
            named_losses[loss_name] = named_losses.get(loss_name, 0) + curr_logit_loss.item()
            if argmax_check(y, y_hat):
                correct_by_act[act_spec] = correct_by_act.get(act_spec, 0) + 1
    return loss, named_losses, correct_by_act 

