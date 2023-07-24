# Alternate implementations of CSVAE, found on Github or adjusted variants

# Trial implementation from https://github.com/nikkou/latent-subspaces
# Seems to be relatively correct, minus the W and Z error
#
# This is set to operate over images
#

import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists
import torch.utils.data as utils
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import torchvision.utils as vutils

from enum import IntEnum

class LossFnEnum(IntEnum):
    MSE = 0
    MULTINOMIAL = 1

# Original values used in the repo.  The extremely high beta1 is
# pretty suspect.
beta1 = 10000 # data_term
beta2 = 0.001 # w_kl
beta3 = 0.1 # z_kl
beta4 = 10
beta5 = 1

class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape=shape
    def forward(self,input):
        return input.view(self.shape)
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)
    
class Conv_block(nn.Module):
    def __init__(self, num_features, in_channels, out_channels, kernel_size, stride=1, padding=0, negative_slope=0.2, p=0.04, transpose=False):
        super(Conv_block, self).__init__()
        
        self.transpose = transpose
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            
        self.activation = nn.LeakyReLU(negative_slope, inplace=True)
        self.dropout = nn.Dropout2d(p)
        #self.batch_norm = nn.BatchNorm2d(num_features)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if not self.transpose:
            x = self.dropout(x)
        #x = self.batch_norm(x)

        return x
    
class CSVAE(nn.Module):
    def __init__(self, labels_dim, z_dim, w_dim, KOF=64, p=0.04, in_channels=3, loss_fn_idx=LossFnEnum.MSE):
        super(CSVAE, self).__init__()
        self.labels_dim = labels_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.in_channels = in_channels
        # Set the loss function, fn(guess, gold)
        self.loss_fn_idx = loss_fn_idx
         
        # x to x_features_dim
        self.encoder = nn.Sequential()
        self.encoder.add_module("block01", Conv_block(KOF, self.in_channels, KOF, 4, 2, 1, p=p))
        self.encoder.add_module("block02", Conv_block(KOF*2, KOF, KOF*2, 4, 2, 1, p=p))
        self.encoder.add_module("block03", Conv_block(KOF*4, KOF*2, KOF*4, 4, 2, 1, p=p))
        self.encoder.add_module("block04", Conv_block(KOF*8, KOF*4, KOF*8, 4, 2, 1, p=p))
        self.encoder.add_module("block05", Conv_block(KOF*16, KOF*8, KOF*16, 4, 2, 1, p=p))
        self.encoder.add_module("block06", Conv_block(KOF*16, KOF*16, KOF*16, 4, 2, 1, p=p))
#         self.encoder.add_module("block07", Conv_block(KOF*32, KOF*16, KOF*32, 4, 2, 1, p=p))
#         self.encoder.add_module("block08", Conv_block(KOF*32, KOF*32, KOF*32, 4, 2, 1, p=p))
        self.encoder.add_module("flatten", Flatten())
    
        x_features_dim = KOF * 8 * 2
        
        self.encoder_xy_to_w = nn.Sequential(
            nn.Linear(x_features_dim + labels_dim, w_dim), 
            nn.ReLU(), 
        )
        self.mu_xy_to_w = nn.Linear(w_dim, w_dim)
        self.logvar_xy_to_w = nn.Linear(w_dim, w_dim)
        
        self.encoder_x_to_z = nn.Sequential(
            nn.Linear(x_features_dim, z_dim), 
            nn.ReLU(), 
        )
        self.mu_x_to_z = nn.Linear(z_dim, z_dim)
        self.logvar_x_to_z = nn.Linear(z_dim, z_dim)
        
        self.encoder_y_to_w = nn.Sequential(
            nn.Linear(labels_dim, w_dim), 
            nn.ReLU(), 
#             nn.Linear(w_dim, w_dim), 
#             nn.ReLU()
        )
        self.mu_y_to_w = nn.Linear(w_dim, w_dim)
        self.logvar_y_to_w = nn.Linear(w_dim, w_dim)
        
        # Add sigmoid or smth for images!
        # (z+w) to x_sample
        # (!) no logvar for x
        self.decoder_zw_to_x = nn.Sequential()
        self.decoder_zw_to_x.add_module("block00", nn.Sequential(
            nn.Linear(z_dim+w_dim, z_dim+w_dim), 
            # nn.BatchNorm1d(z_dim+w_dim), 
            nn.LeakyReLU(0.2)
        ))
        self.decoder_zw_to_x.add_module("reshape", Reshape((-1, z_dim+w_dim, 1, 1)))
        
        self.decoder_zw_to_x.add_module("block01", Conv_block(KOF*4, z_dim+w_dim, KOF*4, 4, 1, 0, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block02", Conv_block(KOF*4, KOF*4, KOF*4, 4, 2, 1, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block03", Conv_block(KOF*2, KOF*4, KOF*2, 3, 1, 1, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block04", Conv_block(KOF*2, KOF*2, KOF*2, 4, 2, 1, p=p, transpose=True))
#         self.decoder_zw_to_x.add_module("block05", Conv_block(KOF*4, KOF*4, KOF*4, 4, 2, 1, p=p, transpose=True))
#         self.decoder_zw_to_x.add_module("block06", Conv_block(KOF*2, KOF*4, KOF*2, 4, 2, 1, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block05", Conv_block(KOF, KOF*2, KOF, 4, 2, 1, p=p, transpose=True))
    # Set padding to 2 if multinomial
#        self.decoder_zw_to_x.add_module("block06", Conv_block(KOF, KOF, KOF, 4, 2, 2, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block06", Conv_block(KOF, KOF, KOF, 4, 2, 1, p=p, transpose=True))

#         self.decoder_zw_to_x.add_module("block07", nn.Sequential(
#                     nn.ConvTranspose2d(KOF, 3, 3, 1, 1)))

        self.mu_zw_to_x = nn.Sequential(
            nn.ConvTranspose2d(KOF, self.in_channels, self.in_channels, 1, 1),
            nn.Tanh()
        )
        self.logvar_zw_to_x = nn.Sequential(
            nn.ConvTranspose2d(KOF, self.in_channels, self.in_channels, 1, 1),
#             nn.Tanh()
        )
#         self.logvar_zw_to_x = nn.Linear(z_dim+w_dim, input_dim)

        # adversarial delta(z -> y)
        self.decoder_z_to_y = nn.Sequential(
            nn.Linear(z_dim, z_dim), 
            nn.ReLU(), 
            nn.Linear(z_dim, z_dim), 
            nn.ReLU(),
            nn.Linear(z_dim, labels_dim), 
            nn.Sigmoid()
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
        
    def q_zw(self, x, y):
        """
        VARIATIONAL POSTERIOR
        :param x: input image
        :return: parameters of q(z|x), (MB, hid_dim)
        """

        x_features = self.encoder(x)
        
        intermediate = self.encoder_x_to_z(x_features)
        z_mu = self.mu_x_to_z(intermediate)
        z_logvar = self.logvar_x_to_z(intermediate)
        
        xy = torch.cat([x_features, y], dim=1)
        
        intermediate = self.encoder_xy_to_w(xy)
        w_mu_encoder = self.mu_xy_to_w(intermediate)
        w_logvar_encoder = self.logvar_xy_to_w(intermediate)
        
        intermediate = self.encoder_y_to_w(y)
        w_mu_prior = self.mu_y_to_w(intermediate)
        w_logvar_prior = self.logvar_y_to_w(intermediate)
        
        return w_mu_encoder, w_logvar_encoder, w_mu_prior, \
               w_logvar_prior, z_mu, z_logvar
    
    def p_x(self, z, w):
        """
        GENERATIVE DISTRIBUTION
        :param z: latent vector          (MB, hid_dim)
        :return: parameters of p(x|z)    (MB, inp_dim)
        """
        
        zw = torch.cat([z, w], dim=1)
        
        intermediate = self.decoder_zw_to_x(zw)
        mu = self.mu_zw_to_x(intermediate)
        logvar = self.logvar_zw_to_x(intermediate)
        
        return mu, logvar

    def forward(self, x, y):
        """
        Encode the image, sample z and decode 
        :param x: input image
        :return: parameters of p(x|z_hat), z_hat, parameters of q(z|x)
        """
        w_mu_encoder, w_logvar_encoder, w_mu_prior, \
            w_logvar_prior, z_mu, z_logvar = self.q_zw(x, y)
        w_encoder = self.reparameterize(w_mu_encoder, w_logvar_encoder)
        w_prior = self.reparameterize(w_mu_prior, w_logvar_prior)
        
        z = self.reparameterize(z_mu, z_logvar)
        x_mu, x_logvar = self.p_x(z, w_encoder)
        
        zw = torch.cat([z, w_encoder], dim=1) # for adversarial train
        
        y_pred = self.decoder_z_to_y(z) # for adversarial train
        
        return x_mu, x_logvar, zw, y_pred, \
               w_mu_encoder, w_logvar_encoder, w_mu_prior, \
               w_logvar_prior, z_mu, z_logvar

    def reconstruct_x(self, x, y):
        x_mu, x_logvar, zw, y_pred, w_mu_encoder, w_logvar_encoder, w_mu_prior, w_logvar_prior, z_mu, z_logvar = self.forward(x, y)
        return x_mu

    def calculate_loss(self, x, y, unit_labels,
                       beta1=20, beta2=1, beta3=0.2, beta4=10, beta5=1):
        """
        Given the input batch, compute the negative ELBO 
        :param x:   (MB, inp_dim)
        :param beta: Float
        :param average: Compute average over mini batch or not, bool
        :return: -RE + beta * KL  (MB, ) or (1, )
        """
        x_mu, x_logvar, zw, y_pred, w_mu_encoder, w_logvar_encoder, w_mu_prior, w_logvar_prior, z_mu, z_logvar = self.forward(x, y)
    
        if self.loss_fn_idx == LossFnEnum.MSE:
            x_recon = nn.MSELoss()(x_mu, x)
        elif self.loss_fn_idx == LossFnEnum.MULTINOMIAL:
            self.loss_fn = multinomial_loss_fn(x_mu, unit_labels)
        else:
            print("ERROR, unknown loss function idx!")
        
        # w_kl
        w_dist = dists.MultivariateNormal(w_mu_encoder.flatten(), 
                                          torch.diag(w_logvar_encoder.flatten().exp()))
        w_prior = dists.MultivariateNormal(w_mu_prior.flatten(), 
                                           torch.diag(w_logvar_prior.flatten().exp()))
        w_kl = dists.kl.kl_divergence(w_dist, w_prior)
        
        # z_kl
        z_dist = dists.MultivariateNormal(z_mu.flatten(), 
                                          torch.diag(z_logvar.flatten().exp()))
        z_prior = dists.MultivariateNormal(torch.zeros(self.z_dim * z_mu.size()[0]).to(z_mu), 
                                           torch.eye(self.z_dim * z_mu.size()[0]).to(z_mu))
        z_kl = dists.kl.kl_divergence(z_dist, z_prior)
        
        # -H(y)
        y_pred_negentropy = (y_pred.log() * y_pred + (1-y_pred).log() * (1-y_pred)).mean()

        # y xentropy
        y_recon = nn.BCELoss()(y_pred, y)        # alternatively use predicted logvar too to evaluate density of input
        
        # total
        ELBO = beta1 * x_recon + beta2 * w_kl + beta3 * z_kl + beta4 * y_pred_negentropy
        return ELBO, x_recon, w_kl, z_kl, y_pred_negentropy, y_recon
        
#         if average:
#             ELBO = ELBO.mean()
#             recon = recon.mean()
#             z_kl = z_kl.mean()
#             w_kl = w_kl.mean()

#         return ELBO, recon, z_kl, w_kl


    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)
    
    
def train():
    model = CSVAE(labels_dim=N_COND, z_dim=64, w_dim=64, KOF=64, p=0.04, in_channels=metadata.N_DISTINCT_UNITS).to(device)
    model.train()

    params_without_delta = [param for name, param in model.named_parameters() if 'decoder_z_to_y' not in name]
    opt_without_delta = optim.Adam(params_without_delta, lr=lr)
    scheduler_without_delta = optim.lr_scheduler.MultiStepLR(opt_without_delta, milestones=[pow(3, i) for i in range(7)], gamma=pow(0.1, 1/7))

    params_delta = [param for name, param in model.named_parameters() if 'decoder_z_to_y' in name]
    opt_delta = optim.Adam(params_delta, lr=lr)
    scheduler_delta = optim.lr_scheduler.MultiStepLR(opt_delta, milestones=[pow(3, i) for i in range(7)], gamma=pow(0.1, 1/7))

    n_epochs = max_epochs
    
    x_recon_losses = []
    w_kl_losses = []
    z_kl_losses = []
    y_negentropy_losses = []
    y_recon_losses = []
    for epoch_i in range(n_epochs):
        for cur_data, unit_labels, cur_attr in tqdm(train_dataloader):
            cur_data = torch.tensor(cur_data, dtype=torch.float32).to(device)
            cur_attr = torch.tensor(cur_attr, dtype=torch.float32).to(device)
            unit_labels = torch.tensor(unit_labels, dtype=torch.long).to(device)

            loss_val, x_recon_loss_val, w_kl_loss_val, z_kl_loss_val, \
            y_negentropy_loss_val, y_recon_loss_val = model.calculate_loss(
                cur_data, cur_attr, unit_labels, beta1=beta1, beta2=beta2, beta3=beta3, beta4=beta4, beta5=beta5)

            # optimization could be done more precisely but less efficiently by only updating delta or other params on a batch

            opt_delta.zero_grad()
            y_recon_loss_val.backward(retain_graph=True)
            opt_delta.step()

            opt_without_delta.zero_grad()
            loss_val.backward()
            opt_without_delta.step()

            x_recon_losses.append(x_recon_loss_val.item())
            w_kl_losses.append(w_kl_loss_val.item())
            z_kl_losses.append(z_kl_loss_val.item())
            y_negentropy_losses.append(y_negentropy_loss_val.item())
            y_recon_losses.append(y_recon_loss_val.item())
        scheduler_without_delta.step()
        scheduler_delta.step()

        #clear_output(True)
        #plot_results(epoch_i)
        print('Epoch {}'.format(epoch_i))
        mean_x_recon = np.array(x_recon_losses[-len(train_dataloader):]).mean()
        mean_z_kl = np.array(z_kl_losses[-len(train_dataloader):]).mean()
        mean_w_kl = np.array(w_kl_losses[-len(train_dataloader):]).mean()
        mean_y_negentropy = np.array(y_negentropy_losses[-len(train_dataloader):]).mean()
        mean_y_recon = np.array(y_recon_losses[-len(train_dataloader):]).mean()
        print('Mean MSE(X): {:.4f}, scaled MSE(X): {:.4f}'.format(mean_x_recon, beta1 * mean_x_recon))
        print('Mean KL W: {:.4f}, scaled KL W: {:.4f}'.format(mean_w_kl, beta2 * mean_w_kl))
        print('Mean KL Z: {:.4f}, scaled KL Z: {:.4f}'.format(mean_z_kl, beta3 * mean_z_kl))
        print('Mean -H(y): {:.4f}, scaled -H(y): {:.4f}'.format(mean_y_negentropy, beta4 * mean_y_negentropy))
        print('Mean BCE(y): {:.4f}, scaled BCE(y): {:.4f}'.format(mean_y_recon, beta5 * mean_y_recon)) # (same)
        print()
