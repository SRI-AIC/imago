import torch
import torch.nn as nn
import torch.nn.functional as tfunc
import collections

from imago_prev.models.sequential.dsv.dis_seq_vae import reparameterize, init_weights

from absl import logging, flags
FLAGS = flags.FLAGS

"""
Implements a World Model, in the form of a DSV without the temporally invariant prior
"""

class DynamicEncoder(nn.Module):
    """
    Implements q(z|x) = p(z|x)
    """
    def __init__(self, input_size, hidden_size, latent_size, dropout=0):
        super(DynamicEncoder, self).__init__()
        self.input_size, self.hidden_size, self.latent_size = input_size, hidden_size, latent_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.rnn = nn.RNN(input_size=hidden_size * 2, hidden_size=hidden_size,
                          batch_first=True)
        self.linear_mu = nn.Linear(hidden_size, latent_size)
        self.linear_logvar = nn.Linear(hidden_size, latent_size)
        self.dropout=dropout
        if self.dropout > 0:
            self.drop_mean = nn.Dropout(p=dropout)
            self.drop_logvar = nn.Dropout(p=dropout)
        else:
            self.drop_mean, self.drop_logvar = None

    def forward(self, X):
        X, _ = self.lstm(X)
        X, _ = self.rnn(X)
        if self.dropout > 0:
            mu = self.linear_mu(self.drop_mean(X))
            logvar = self.linear_logvar(self.drop_logvar(X))
        else:
            mu = self.linear_mu(X)
            logvar = self.linear_logvar(X)
        return mu, logvar

    