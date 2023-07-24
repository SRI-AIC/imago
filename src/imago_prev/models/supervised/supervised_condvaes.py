import os

import numpy as np
import random
from torch.utils.data import Dataset
from absl import logging
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from tqdm import tqdm
from sc2recorder.utils import ensure_numpy, ensure_torch, ensure_torch_long
from imago_prev.models.multinomial_vaes import *
from imago_prev.models.multinomial_vaes import multinomial_loss_fn
from imago_prev.models.model_util import save_checkpoint


def train_model(model, optimizer, train_dataloader, test_dataloader, metadata,
                epochs=10, device="cpu",
                save_model_dir="output/model",
                save_model_epochs=100,
                kl_penalty_regime="full"):
    """
    Trains the Conditional VAE model, with only interpolation multinomial reconstruction loss.
    The train and test dataloaders must be backed by datasets that can index
    off of conditioning variables.
    :param model:
    :param optimizer:
    :param train_dataloader:
    :param test_dataloader:
    :param metadata:
    :param epochs:
    :param device:
    :param save_model_dir:
    :param save_model_epochs:
    :param kl_penalty_regime:
    :return:
    """
    for epoch in range(epochs):
        epoch_loss = 0.
        optimizer.zero_grad()
        tqdm_iter = tqdm(train_dataloader)
        for X1, Y1, C1, X2, Y2, C_d in tqdm_iter:
            (X1, C1, X2, C_d) = ensure_torch(device, X1, C1, X2, C_d)
            Y1, Y2 = ensure_torch_long(device, Y1, Y2)
            optimizer.zero_grad()
            X2_hat, mu2, logvar2, z2 = model.forward_interpolate(X1, C1, C_d)
            loss = multinomial_loss_fn(X2_hat, Y2)
            step_loss = loss.item()
            epoch_loss += step_loss
            tqdm_iter.set_description("Epoch {}, Step Loss={:.5f}, Epoch Loss={:.5f}".format(epoch, step_loss, epoch_loss))
            loss.backward()
            optimizer.step()


def train_model_2head_old(model, optimizer, train_dataloader, test_dataloader, metadata,
                epochs=10, device="cpu",
                save_model_dir="output/model",
                save_model_epochs=100,
                kl_penalty_regime="no_kl"):
    """
    Trains the Conditional VAE model, with an additional multinomial reconstruction loss at the
    interpolation.  The train and test dataloaders must be backed by datasets that can index
    off of conditioning variables.
    :param model:
    :param optimizer:
    :param train_dataloader:
    :param test_dataloader:
    :param metadata:
    :param epochs:
    :param device:
    :param save_model_dir:
    :param save_model_epochs:
    :param kl_penalty_regime:
    :return:
    """
    logging.info("Training, size train={}, size valid={}".format(len(train_dataloader), len(test_dataloader)))
    for epoch in range(epochs):
        epoch_loss = 0.
        tqdm_iter = tqdm(train_dataloader)
        for X1, Y1, C1, X2, Y2, C_d in tqdm_iter:
            (X1, C1, X2, C_d) = ensure_torch(device, X1, C1, X2, C_d)
            Y1, Y2 = ensure_torch_long(device, Y1, Y2)
            optimizer.zero_grad()
            X1_hat, mu1, logvar1, z1 = model.forward_interpolate(X1, C1, C1) # Old version, mistakenly put same vector in 
            X2_hat, mu2, logvar2, z2 = model.forward_interpolate(X1, C1, C_d)
            loss = multinomial_loss_fn(X2_hat, Y2)
            loss += multinomial_loss_fn(X1_hat, Y1)
            if kl_penalty_regime == "no_kl":
                pass
            else:
                # Default, full KL
                loss += torch.mean(0.5 * torch.sum(torch.exp(logvar1) + mu1**2 - 1. - logvar1, 1))  # KL divergence term, closed form
                loss += torch.mean(0.5 * torch.sum(torch.exp(logvar2) + mu2**2 - 1. - logvar2, 1))  # KL divergence term, closed form
            step_loss = loss.item()
            epoch_loss += step_loss
            tqdm_iter.set_description("Epoch {}, Step Loss={:.5f}, Epoch Loss={:.5f}".format(epoch, step_loss, epoch_loss))
            loss.backward()
            optimizer.step()
        valid_recon_loss = 0.
        valid_interp_loss = 0.
        recon_kl_loss, interp_kl_loss = 0., 0.
        valid_tqdm_iter = tqdm(test_dataloader)
        for Xt1, Yt1, Ct1, Xt2, Yt2, C_dt in valid_tqdm_iter:
            Xt1, Ct1, Xt2, C_dt = ensure_torch(device, Xt1, Ct1, Xt2, C_dt)
            Yt1, Yt2 = ensure_torch_long(device, Yt1, Yt2)
            Xt1_hat, mu1_t, logvar1_t, z1_t = model.forward_interpolate(Xt1, Ct1, Ct1)  # Old version, mistakenly put same vector in
            Xt2_hat, mu2_t, logvar2_t, z2_t = model.forward_interpolate(Xt2, Ct1, C_dt)
            valid_recon_loss += multinomial_loss_fn(Xt1_hat, Yt1).item()
            valid_interp_loss += multinomial_loss_fn(Xt2_hat, Yt2).item()
            if kl_penalty_regime == "full":
                recon_kl_loss = torch.mean(
                    0.5 * torch.sum(torch.exp(logvar1) + mu1 ** 2 - 1. - logvar1, 1))  # KL divergence term, closed form
                interp_kl_loss = torch.mean(
                    0.5 * torch.sum(torch.exp(logvar2) + mu2 ** 2 - 1. - logvar2, 1))  # KL divergence term, closed form
            valid_tqdm_iter.set_description("Validation: Epoch {}, Recon Loss={:.5f}/KL={:.5f}, Interp. Loss={:.5f}/KL={:.5f}".format(epoch, valid_recon_loss, recon_kl_loss, valid_interp_loss, interp_kl_loss))


def gen_reconstruction_imgs(model, X, C, renderer, device):
    X_hat, mu, logvar, z = model.forward_interpolate(X, C, torch.zeros(C.size()).to(device))
    orig_imgs = [np.array(img) for img in renderer.display(X)]
    reconst_imgs = [np.array(img) for img in renderer.display(X_hat)]
    paired = []
    for orig, reconst in zip(orig_imgs, reconst_imgs):
        paired.append(orig)
        paired.append(reconst)
    return np.array(paired)

def gen_interp_imgs(model, X, C, C_d, renderer):
    X_hat, mu, logvar, z = model.forward_interpolate(X, C, C_d)
    orig_imgs = [np.array(img) for img in renderer.display(X)]
    reconst_imgs = [np.array(img) for img in renderer.display(X_hat)]
    paired = []
    for orig, reconst in zip(orig_imgs, reconst_imgs):
        paired.append(orig)
        paired.append(reconst)
    return np.array(paired)


def train_model_2head(model, optimizer, train_dataloader, test_dataloader,
                      renderer,
                    epochs=10, device="cpu",
                    save_model_dir="output/model",
                    save_model_epochs=100,
                    save_img_steps=10000,
                    kl_penalty_regime="full",
                    kl_penalty_weight=1,
                    activate_recon_loss=True,
                    activate_interp_loss=True,
                    recon_decode_full_C=False,
                    interp_decode_full_C=False,
                      **kwargs):
    """
    Trains the Conditional VAE model, with an additional multinomial reconstruction loss at the
    interpolation.  The train and test dataloaders must be backed by datasets that can index
    off of conditioning variables.
    :param model:
    :param optimizer:
    :param train_dataloader:
    :param test_dataloader:
    :param metadata:
    :param epochs:
    :param device:
    :param save_model_dir:
    :param save_model_epochs:
    :param kl_penalty_regime:
    :return:
    """
    writer = SummaryWriter(log_dir=os.path.join(save_model_dir, "logs"))
    logging.info("Training, size train={}, size valid={}".format(len(train_dataloader), len(test_dataloader)))
    steps = 0
    for epoch in range(epochs):
        epoch_loss = 0.
        tqdm_iter = tqdm(train_dataloader)
        for X1, Y1, C1, X2, Y2, C_d in tqdm_iter:
            steps += 1
            (X1, C1, X2, C_d) = ensure_torch(device, X1, C1, X2, C_d)
            Y1, Y2 = ensure_torch_long(device, Y1, Y2)
            optimizer.zero_grad()
            if recon_decode_full_C:
                X1_hat, mu1, logvar1, z1 = model.forward_interpolate(X1, C1, C1)
            else:
                X1_hat, mu1, logvar1, z1 = model.forward_interpolate(X1, C1, torch.zeros(C_d.size()).to(device))
            if interp_decode_full_C:
                X2_hat, _, _, _ = model.forward_interpolate(X1, C1, C1 + C_d)
            else:
                X2_hat, _, _, _ = model.forward_interpolate(X1, C1, C_d)
            loss = 0.
            if activate_recon_loss:
                recon_loss = multinomial_loss_fn(X1_hat, Y1)
                loss += recon_loss
                writer.add_scalar("Reconst_loss", recon_loss.item(), steps)
            if activate_interp_loss:
                interp_loss = multinomial_loss_fn(X2_hat, Y2)
                writer.add_scalar("Interp_loss", interp_loss.item(), steps)
                loss += interp_loss
            kl1_loss, kl2_loss = 0., 0.
            if kl_penalty_regime == "full":
                # Only makes sense to pnealize KL once, since interp and recon share the same
                # variables.
                if activate_recon_loss or activate_interp_loss:
                    kl1_loss = kl_penalty_weight * (-0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp()))
                    loss += kl1_loss
                    writer.add_scalar("kl_recon_loss", kl1_loss.item(), steps)
            step_loss = loss.item()
            writer.add_scalar("step_loss", step_loss, steps)
            epoch_loss += step_loss
            tqdm_iter.set_description("Epoch {}, Step Loss={:.5f}, KL_losses={:.5f}/{:.5f} Epoch Loss={:.5f}".format(epoch, step_loss, kl1_loss, kl2_loss, epoch_loss))
            loss.backward()
            optimizer.step()
            
        if activate_recon_loss:
            train_reconst_imgs = gen_reconstruction_imgs(model, X1, C1, renderer, device)
            writer.add_images("Train_Reconst", train_reconst_imgs, global_step=steps, dataformats='NHWC')
        if activate_interp_loss:
            train_interp_imgs = gen_interp_imgs(model, X1, C1, C_d, renderer)
            writer.add_images("Train_Interp", train_interp_imgs, global_step=steps, dataformats='NHWC')

        if epoch % save_model_epochs == 0:
            save_checkpoint("model_{}".format(epoch), model, optimizer, save_model_dir, epoch)
        valid_recon_loss = 0.
        valid_interp_loss = 0.
        recon_kl_loss, interp_kl_loss = 0., 0.
        valid_tqdm_iter = tqdm(test_dataloader)
        for Xt1, Yt1, Ct1, Xt2, Yt2, C_dt in valid_tqdm_iter:
            Xt1, Ct1, Xt2, C_dt = ensure_torch(device, Xt1, Ct1, Xt2, C_dt)
            Yt1, Yt2 = ensure_torch_long(device, Yt1, Yt2)
            if recon_decode_full_C:
                Xt1_hat, mu1_t, logvar1_t, z1_t = model.forward_interpolate(Xt1, Ct1, C1)
            else:
                Xt1_hat, mu1_t, logvar1_t, z1_t = model.forward_interpolate(Xt1, Ct1, torch.zeros(C_dt.size()).to(device))
            if interp_decode_full_C:
                Xt2_hat, _, _, _ = model.forward_interpolate(Xt2, Ct1, C1 + C_dt)
            else:
                Xt2_hat, _, _, _ = model.forward_interpolate(Xt2, Ct1, C_dt)
            valid_recon_loss += multinomial_loss_fn(Xt1_hat, Yt1).item()
            valid_interp_loss += multinomial_loss_fn(Xt2_hat, Yt2).item()
            if kl_penalty_regime == "full":
                recon_kl_loss = kl_penalty_weight * (-0.5 * torch.sum(1 + 2 * logvar1_t - mu1_t.pow(2) - (2 * logvar1_t).exp())).item()
            valid_tqdm_iter.set_description("Validation: Epoch {}, Recon Loss={:.5f}/KL={:.5f}, Interp. Loss={:.5f}".format(epoch, valid_recon_loss, recon_kl_loss, valid_interp_loss))

            if activate_recon_loss:
                writer.add_scalar("Valid_reconst_loss", valid_recon_loss, steps)
                writer.add_scalar("Valid_recon_KL_loss", recon_kl_loss, steps)
                valid_reconst_imgs = gen_reconstruction_imgs(model, Xt1, Ct1, renderer, device)
                writer.add_images("Valid_Reconst", valid_reconst_imgs, global_step=steps, dataformats='NHWC')
            if activate_interp_loss:
                writer.add_scalar("Valid_interp_loss", valid_interp_loss, steps)
                valid_interp_imgs = gen_interp_imgs(model, Xt1, Ct1, C_dt, renderer)
                writer.add_images("Valid_Interp", valid_interp_imgs, global_step=steps, dataformats='NHWC')


    save_checkpoint("model_{}".format(epoch), model, optimizer, save_model_dir, epoch)