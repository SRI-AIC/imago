import torch
import torch.nn as nn

from imago.models.encdec_schedules import get_encoder_schedules, get_z_schedules, get_decoder_schedules

class MultiInputConvVAE(nn.Module):
    """
    This is a Convolutional VAE that accepts an input tensor sized (F, H, W, C).
    This passes through a convolutional decoder then encoder, leaving
    a spatial layer that is further interpreted by MultiConvOutput
    heads from the main model.

    Following the intuition of StyleGAN, adds an affine transformation.

    Here we use a dense latent approach, encoding at each conv layer and then
    concatenating them into a single latent.

    NOTE: Currently hardcoded to 64x64
    """

    def __init__(self, NUM_INPUTS, N_TARGETS,
                 N_LATENT=256, N_HIDDEN=32, height=64, width=64,
                 nc=32):
        super(MultiInputConvVAE, self).__init__()
        self.WIDTH, self.HEIGHT = height, width
        self.N_LATENT, self.N_HIDDEN = N_LATENT, N_HIDDEN

        # Note: This is arrayed so each of the four encoding
        # layers generate an encoding sized 128, so 128 * 4 = 512 = 2 * 256
        # This is currently hardwired to defaults, including the encoder
        # schedules.
        # TODO: Make this more modular!
        # Hack to get this to work with Cartpole; need to make this more robust in the future
        if height == 1 and width == 1:
            self.COMBINED_Z_DIM = 64
            self.pre_dec_shape = (1, 1)
        else:
            # Standard 2D convolution.  pre_dec shape is the 'seed' shape used to
            # generate the spatial outputs
            self.COMBINED_Z_DIM = 2 * N_LATENT
            self.pre_dec_shape = (2, 2)

        encoding_schedule = get_encoder_schedules(NUM_INPUTS, N_HIDDEN, height, width)
        self.encoder_list = nn.ModuleList(
            [ConvZ(c.num_in, c.num_out,
                   kernel_size=c.kernel_size,
                   stride=c.stride,
                   padding=c.padding,
                   dilation=c.dilation) for c in encoding_schedule]
            )

        enc_z_schedule = get_z_schedules(nc, nc, height, width)
        self.z_encoders = []
        # Construct the multi-level input to do the flattened
        # hierarchical representation for Z.  This is really only
        # relevant for convolutional architectures, where
        # information is abstracted at each level.
        for start_idx in range(0, len(enc_z_schedule)):
            z_encoder = []
            for i in range(start_idx, len(enc_z_schedule)):
                c = enc_z_schedule[i]
                z_encoder.append(ConvZ(c.num_in, c.num_out, kernel_size=c.kernel_size,
                                       stride=c.stride, padding=c.padding, dilation=c.dilation))
            z_encoder = nn.Sequential(*z_encoder)
            self.z_encoders.append(z_encoder)
        self.z_encoders = nn.ModuleList(self.z_encoders)

        self.z2mu = nn.Linear(self.COMBINED_Z_DIM, N_LATENT)
        self.z2logvar = nn.Linear(self.COMBINED_Z_DIM, N_LATENT)

        # Project latent into a spatial form, for transpose convolution upsampling
        self.fc3 = nn.Linear(N_LATENT,
                             N_LATENT * self.pre_dec_shape[0] * self.pre_dec_shape[1])  # TODO: determine shape dynamically

        decoder_schedule = get_decoder_schedules(height, width, n_targets=N_TARGETS,
                                                 n_latent=N_LATENT, n_hidden=N_HIDDEN)
        dec_layers = [TranspConvZ(c.num_in, c.num_out, kernel_size=c.kernel_size, stride=c.stride)
                      for c in decoder_schedule]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, X):
        layerwise_res = []
        layerwise_Z = []
        batch_size = X.shape[0]

        # Go through each step in the encoding, forking off
        # a separate layer-specific Z for eventual concatenation
        # and gisting at the end
        for idx in range(len(self.encoder_list)):
            encoder_layer = self.encoder_list[idx]
            X = encoder_layer(X)
            layerwise_res.append(X)
            if idx < len(self.z_encoders):
                layer_Z = self.z_encoders[idx](X)
                layerwise_Z.append(layer_Z.reshape(batch_size, -1))
            else:
                layerwise_Z.append(X.reshape(batch_size, -1))
        combined_Z = torch.cat(layerwise_Z, 1)
        mu = self.z2mu(combined_Z)
        logvar = self.z2logvar(combined_Z)
        return mu, logvar

    def is_conditional(self):
        return True

    #    def encode(self, x):
    #        h1 = self.encoder(x)
    #        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
            return z
        else:
            return mu

    def decode(self, a):
        h3 = self.fc3(a)
        h3 = h3.view(-1, self.N_LATENT, self.pre_dec_shape[0], self.pre_dec_shape[1])
        return self.decoder(h3)

    def forward(self, x, deterministic=False):
        """ Similar to regular forward, but passes the z through an affine translation layer """
        mu, logvar = self.encode(x)
        if deterministic:
            z_pre = mu
        else:
            z_pre = self.reparameterize(mu, logvar)
        return self.decode(z_pre), mu, logvar, z_pre

    def forward_Z(self, z):
        """ Similar to regular forward, but passes the z through an affine translation layer """
        return self.decode(z)

    def embed(self, x, reparameterize=False):
        mu, logvar = self.encode(x)
        if reparameterize:
            z_pre = self.reparameterize(mu, logvar)
            return self.aff(self.mapper(z_pre))
        else:
            return self.aff(self.mapper(mu))

    def map_affine(self, z_latent):
        return self.aff(self.mapper(z_latent))

    def forward_det(self, x):
        """ Deterministic, with no resampling (just use means)
        """
        mu, logvar = self.encode(x)
        z = mu
        return self.decode(a), mu, logvar, self.aff(self.mapper(z))

#
# Components for ConvVAE
#

class ConvZ(nn.Module):
    """
    Encoder component: Has the Convolutional2D portion, but also a latent output
    """

    def __init__(self, num_c_in, num_c_out, kernel_size=4, stride=2,
                 padding=0, dilation=1):
        super(ConvZ, self).__init__()
        self.num_c_in, self.num_c_out = num_c_in, num_c_out
        self.model = nn.Sequential(
            nn.Conv2d(num_c_in, num_c_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation),
            nn.BatchNorm2d(num_c_out),
            nn.ReLU())

    def forward(self, X):
        return self.model(X)


class TranspConvZ(nn.Module):
    """ Transpose convolution.  Accepts 2 * latent_size inputs on decode"""

    def __init__(self, num_in, num_out,
                 kernel_size=4, stride=2):
        super(TranspConvZ, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(num_in, num_out, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(num_out),
            nn.ReLU())

    def forward(self, X):
        Y = self.model(X)
        return Y
