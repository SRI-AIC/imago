import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPVAE(nn.Module):
    def __init__(self, latent2targets=[],
                 Z_DIM=2, FC1_DIM=4):
        super(MLPVAE, self).__init__()
        self.latent2targets = latent2targets
        self.encoder = nn.Sequential(
            nn.Linear(4, FC1_DIM),
            nn.ReLU(),
            nn.Linear(FC1_DIM, Z_DIM),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(Z_DIM,Z_DIM)
        self.fc_logvar = nn.Linear(Z_DIM,Z_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(Z_DIM, FC1_DIM),
            nn.ReLU(),
            nn.Linear(FC1_DIM, 4),
        )
        targets = collections.OrderedDict()
        for l2t in self.latent2targets:
            name = l2t.name
            targets[name] = \
                nn.Sequential(
                    nn.Linear(Z_DIM, FC1_DIM),
                    nn.ReLU(),
                    nn.Linear(FC1_DIM, 1)
                )
        self.targets = nn.ModuleDict(targets)

    def forward(self, X):
        X1 = self.encoder(X)
        z_mu = self.fc_mu(X1)
        z_logvar = self.fc_logvar(X1)
        z = self.reparameterize(z_mu, z_logvar)
        Xhat, That = self.forward_Z(z)
        return Xhat, z_mu, z_logvar, z, That

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
            return z
        else:
            return mu

    def forward_Z(self, Z):
        Xhat = self.decoder(Z)
        That = {}
        for var_name, var_fn in self.targets.items():
            That[var_name] = var_fn(Z)
        return Xhat, That


def loss_fn(Xh, X_gold, Z_logvar, Z_mu, Th, T_gold, beta=1e-5):
    loss = F.mse_loss(Xh, X_gold)
    named_losses = { 'recon': loss.item() }
    kl_loss = beta * torch.mean(0.5 * torch.sum(torch.exp(Z_logvar) + Z_mu ** 2 - 1. - Z_logvar, 1))
    loss += kl_loss
    for var_name in T_gold.keys():
        if var_name != 'observation' and var_name in Th:
            t_guess = Th[var_name]
            t_gold = T_gold[var_name]
            t_loss = F.mse_loss(t_guess, t_gold)
            loss += t_loss
            named_losses[var_name] = t_loss.item()
    return loss, named_losses