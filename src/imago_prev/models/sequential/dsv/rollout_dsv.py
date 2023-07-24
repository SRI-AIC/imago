import torch
import torch.nn as nn
import torch.nn.functional as tfunc
import collections

from absl import logging, flags
FLAGS = flags.FLAGS


def reparameterize(mu, logvar, deterministic=False):
    if deterministic:
        return mu
    # Get std-dev (0-1, scale)
    eps = torch.randn_like(logvar)
    std = torch.exp(0.5 * logvar)
    sampled = mu + eps * std
    return sampled

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            logging.debug("{}: Initializing module as BatchNorm".format(m))
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 1)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            logging.debug("{}: Initializing module via Kaiming".format(m))
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
            logging.debug("{}: Default init".format(m))


class DynamicEncoder(nn.Module):
    """
    Implements a sequential encoder that first uses a bidirectional LSTM, followed by a RNN
    over the LSTM.  Why not single direction LSTM over bidirectional LSTM?  Why not just a single
    layer?

    Used to compute the full variational distribution

        q(z,f|x) = p(z|x,f), p(f|x)

    Args:
        input_size: Dim of input to recurrent component (usually cat of X and f)
        hidden_size: Recurrent dimension
        latent_size: Dim of characterizing variables (mean and logvar)
    TODO: Why? Why?
    TODO: GRU versus LSTM
    TODO: Transformer
    """
    def __init__(self, input_size, hidden_size, latent_size, dropout=0):
        super(DynamicEncoder, self).__init__()
        self.input_size, self.hidden_size, self.latent_size = input_size, hidden_size, latent_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.rnn = nn.RNN(input_size=hidden_size * 2, hidden_size=hidden_size,
                          batch_first=True)
        self.linear_mu = nn.Linear(hidden_size, latent_size)     # Decodes RNN hidden state
        self.linear_logvar = nn.Linear(hidden_size, latent_size)
        self.dropout = dropout
        if dropout > 0:
            self.drop_mean = nn.Dropout(p=dropout)
            self.drop_logvar = nn.Dropout(p=dropout)
        else:
            self.drop_mean, self.drop_logvar = None

    def forward(self, X):
        X, lstm_hidden = self.lstm(X)
        X, rnn_hidden = self.rnn(X)
        if self.dropout > 0:
            mu = self.linear_mu(self.drop_mean(X))
            logvar = self.linear_logvar(self.drop_logvar(X))
        else:
            mu = self.linear_mu(X)
            logvar = self.linear_logvar(X)
        return mu, logvar


class StaticEncoder(nn.Module):
    """
    Implements a sequential encoder that first uses a bidirectional LSTM, followed by a linear layer
    to compute a single variable.

    Used to compute p(f|x) for the full variational distribution

        q(z,f|x) = p(z|x,f), p(f|x)

    Args:
        input_size: Dim of input to recurrent component (usually cat of X and f)
        hidden_size: Recurrent dimension
        latent_size: Dim of characterizing variables (mean and logvar)
    TODO: Why? Why no RNN compared with Dynamic?
    TODO: GRU versus LSTM
    TODO: Transformer
    """

    def __init__(self, input_size, hidden_size, latent_size, dropout=0.3):
        super(StaticEncoder, self).__init__()
        self.input_size, self.hidden_size, self.latent_size = input_size, hidden_size, latent_size
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                              num_layers=1, bidirectional=True, batch_first=True)
        self.linear_mu = nn.Linear(hidden_size * 2, latent_size)     # hidden_size * 2 b/c bidirectional
        self.linear_logvar = nn.Linear(hidden_size * 2, latent_size)
        self.dropout = dropout
        if dropout > 0:
            self.drop_mean = nn.Dropout(p=dropout)
            self.drop_logvar = nn.Dropout(p=dropout)
        else:
            self.drop_mean, self.drop_logvar = None

    def forward(self, X):
        X, lstm_hidden = self.bilstm(X)
        recurr_out = X[:, -1]
        if self.dropout > 0:
            mu = self.linear_mu(self.drop_mean(recurr_out))
            logvar = self.linear_logvar(self.drop_logvar(recurr_out))
        else:
            mu = self.linear_mu(recurr_out)
            logvar = self.linear_logvar(recurr_out)
        return mu, logvar


class PerceptualEncoder(nn.Module):
    def __init__(self, x_dim=1024, nc=3, h=64, w=64, dropout=0.4,
                 negative_slope=0.1):
        super(PerceptualEncoder, self).__init__()
        self.dr = dropout
        self.nc = nc
        self.h, self.w = h, w
        self.lr_ns = negative_slope
        self.x_dim = x_dim
        self.conv1 = nn.Conv2d(in_channels=self.nc, out_channels=256, kernel_size=4,
                               stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4,
                               stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.drop2 = nn.Dropout2d(self.dr)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4,
                               stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.drop3 = nn.Dropout2d(self.dr)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4,
                               stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout2d(self.dr)
        self.preconv_dim = 4 * 4 * 256
        self.conv_fc = nn.Linear(self.preconv_dim, x_dim)
        self.bn_fc = nn.BatchNorm1d(x_dim)
        self.drop_fc = nn.Dropout(self.dr)

    def forward(self, X):
        """
        Presumes input is of shape (batch-size, num-frames, nc, h, w)
        :param X:
        :return:
        """
        logging.debug("Perceptual Encoder, X.shape={}".format(X.shape))
        bs, num_frames, c, h, w = X.shape
        X = X.reshape(bs * num_frames, self.nc, self.h, self.w)  # Unroll all images into single batched format
        X = self.conv1(X)
        X = tfunc.leaky_relu(X, negative_slope=self.lr_ns)
        X = self.bn2(self.conv2(X))
        X = self.drop2(tfunc.leaky_relu(X, negative_slope=self.lr_ns))
        X = self.bn3(self.conv3(X))
        X = self.drop3(tfunc.leaky_relu(X, negative_slope=self.lr_ns))
        X = self.bn4(self.conv4(X))
        X = self.drop4(tfunc.leaky_relu(X, negative_slope=self.lr_ns))
        X = X.view(-1, self.preconv_dim)
        X = self.drop_fc(tfunc.leaky_relu(self.bn_fc(self.conv_fc(X)), negative_slope=self.lr_ns))
        return X.view(-1, num_frames, self.x_dim)

class Decoder(nn.Module):
    """
    Implements P(X|Z, f)
    """
    def __init__(self, input_dim, preconv_dim=4*4*256,
                 nc=3, h=64, w=64,
                 dropout=0.4,
                 use_leaky_activation=True):
        super(Decoder, self).__init__()
        self.preconv_shape = (256,4,4)
        self.dropout = dropout
        self.input_dim = input_dim
        self.preconv_dim, self.nc, self.h, self.w = preconv_dim, nc, h, w
        self.use_leaky_activation = use_leaky_activation
        self.deconv_fc=nn.Linear(input_dim, self.preconv_dim) # Linear from input vars to decode dim
        self.deconv_bn = nn.BatchNorm1d(self.preconv_dim)
        self.deconv_fc_drop = nn.Dropout(dropout)
        self.deconv4 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.deconv4_drop = nn.Dropout(dropout)
        self.deconv4_bn = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3_drop = nn.Dropout(dropout)
        self.deconv3_bn = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2_drop = nn.Dropout(dropout)
        self.deconv2_bn = nn.BatchNorm2d(256)
        self.deconv1 = nn.ConvTranspose2d(256, nc, kernel_size=4, stride=2, padding=1)

    def forward(self, X):
        logging.debug("Decoder forward X.shape={}".format(X.shape))
        num_frames = X.shape[1] # (batch_size, num_frames, Z+F)
        X = X.view(-1, self.input_dim)  # Unroll input so we see everything as single 'batch' of Z+F embeddings
        X = self.deconv_bn(self.deconv_fc(X))
        if self.use_leaky_activation:
            X = tfunc.leaky_relu(X, 0.1)
        X = self.deconv_fc_drop(X)
        X = X.view(-1, self.preconv_shape[0],
                   self.preconv_shape[1], self.preconv_shape[2]) # Convert into 4x4 256 channel patch for deconvolution
        X = self.deconv4_bn(self.deconv4(X))
        if self.use_leaky_activation:
            X = tfunc.leaky_relu(X, 0.1)
        X = self.deconv4_drop(X)
        X = self.deconv3_bn(self.deconv3(X))
        if self.use_leaky_activation:
            X = tfunc.leaky_relu(X, 0.1)
        X = self.deconv3_drop(X)
        X = self.deconv2_bn(self.deconv2(X))
        if self.use_leaky_activation:
            X = tfunc.leaky_relu(X, 0.1)
        X = self.deconv2_drop(X)
        X = self.deconv1(X)
        if self.use_leaky_activation:
            X = tfunc.leaky_relu(X, 0.1)
        X = torch.tanh(X)
        X = X.view(-1, num_frames, self.nc, self.h, self.w)
        return X


class DynamicsPrior(nn.Module):
    """
    Implements P(Z_{T+1}|Z_T,f).  This is a separate RNN that we parameterise (train) with sampled f and
    Zs from the training data.  This is a predictive model, so input is the previous Z state.  The first
    input in this sequence is Z=0s.

    TODO: Consider using the hidden state in the Z encoding RNN, as that's wider and may contain more information.
    """
    def __init__(self, z_size, f_size, hidden_size=64):
        super(DynamicsPrior, self).__init__()
        self.z_size, self.f_size = z_size, f_size
        self.rnn = nn.RNN(input_size=(f_size + z_size), hidden_size=hidden_size,
                          batch_first=True)
        self.hidden_size = hidden_size
        self.h2z = nn.Linear(hidden_size, z_size)

        # TODO: Consider actually resampled Z transitions
        #self.z_t1_mu = nn.Linear(hidden_size, z_size)


    def forward(self, ZF, f):
        # Performs inference given the concatenated ZF matrix.  The RNN is used to forward project the new
        # Z_hat.
        logging.debug("\n\nf.shape={}, ZF.shape={}\n\n".format(f.shape, ZF.shape))
        (bs, _) = f.shape
        init_zf = torch.cat((torch.zeros((bs, self.z_size), device=f.device), f), dim=-1).unsqueeze(1)
        X = torch.cat((init_zf, ZF), dim=1) # Observations to feed into RNN (include feed for next)
        logging.debug("\n\ninit_zf={}, ZF={}, X={}\n\n".format(init_zf.shape, ZF.shape, X.shape))

        A, _ = self.rnn(X) # Emit intermediate representation and save just the prediction
        Z_hat = self.h2z(A) # Decode into Z value
        logging.debug("\n\nRollout forward, A.shape={}, Zhat_shape={}\n\n".format(A.shape, Z_hat.shape))
        return Z_hat
    


RDSVResult = collections.namedtuple('RDSVResult', ['X_hat', "F_mu", "F_logvar", "Z_mu", "Z_logvar", "Z", "f",
                                                   "Z_hat"])

RolloutResult = collections.namedtuple('RolloutResult', ['X_next', "Z_next", "f"])

class RolloutDisSeqVAE(nn.Module):
    """ Implements the Disentangled Sequential Autoencoder, Li and Mandt 2018,
    This constructs a fixed N version, per the sprites data.

    Updated to perform rollouts for forward prediction
    """
    def __init__(self,
                 num_frames=8,  # Number of fixed frames for model, use num_frames - 1 to train, last to predict
                 z_dim=32,  # Latent variables
                 f_dim=64,  # Bidirectional LSTM internal encoding.  NOTE: Could replace with Transformer
                 f_state_dim=512,  # Used to encode recurrent state for computing state for f|x_{1..t}
                 z_state_dim=512, # Used to encode recurrent state for computing p(z_1..t|x, f)
                 x_dim=1024  # Perceptual FC, input X for system
                 ):
        super(RolloutDisSeqVAE, self).__init__()
        self.num_frames, self.z_dim, self.f_dim = num_frames, z_dim, f_dim  # Use the number of frames -1 for predictive
        self.x_dim = x_dim
        self.f_state_dim, self.z_state_dim = f_state_dim, z_state_dim

        self.nc, self.h, self.w = 3, 64, 64  # Input dimensions
        self.lr_ns = 0.1 # Leaky ReLU Negative Slope
        self.dr = 0.4 # Dropout rate

        self.f_encoder = StaticEncoder(input_size=x_dim,
                                       hidden_size = f_state_dim,
                                       latent_size = f_dim,
                                       dropout = self.dr)
        self.z_encoder = DynamicEncoder(input_size=x_dim + f_dim,
                                        hidden_size=z_state_dim,
                                        latent_size=z_dim,
                                        dropout = self.dr)
        self.x_encoder = PerceptualEncoder()

        # Decoder
        # NOTE: Seems like Resnet-like decoding may help?
        self.decoder = Decoder(input_dim = f_dim + z_dim, dropout=self.dr,
                               nc=self.nc, h=self.h, w=self.w)

        # Stochastic rollout
        self.dynamics_prior = DynamicsPrior(z_dim, f_dim, hidden_size=z_dim*2)

        init_weights(self)
        nn.init.xavier_normal_(self.decoder.deconv1.weight, nn.init.calculate_gain('tanh'))

    def forward(self, full_O):
        """
        Encodes the observations and returns the DSVResult

        Uses 0:nframes-1 frames for the recurrent component, uses the last frame as the
        predictive target.
        :param O: input obs, shape=(nbatch, nframes, C, H, W)
        :return: DSVResult object
        """
        (bs, nf, c, h, w) = full_O.shape
        X = self.x_encoder(full_O)  # Encode obs frames individually
        F_mu, F_logvar = self.f_encoder(X)  # Encode prior using FC embeddings
        f = reparameterize(F_mu, F_logvar)  # Sample it
        F = f.unsqueeze(1).expand(-1, self.num_frames, self.f_dim) # Duplicate F for each X

        Z_mu, Z_logvar = self.z_encoder(torch.cat((X, F), dim=-1))  # Parameters for hidden layer
        Z = reparameterize(Z_mu, Z_logvar)
        ZF = torch.cat((Z,F), dim=-1)

        logging.debug("\n\nRDSVE forward, Z={}, F={}, ZF={}\n\n".format(Z.shape, F.shape, ZF.shape))
        X_hat = self.decoder(ZF)
        Z_hat = self.dynamics_prior(ZF[:, 0:-1, :], f) # Predict the last Z

        return RDSVResult(
            X_hat=X_hat,
            F_mu=F_mu, F_logvar=F_logvar,
            Z_mu=Z_mu, Z_logvar=Z_logvar,
            Z=Z, f=f, Z_hat=Z_hat)

    def sample_rollout(self, O, n=1, f=None, Z_context=None, Z_next=None):
        """ Given a full num_frames input, characterizes using 0:-n frames, and then
        uses the DynamicsPrior to deterministically generate n frames after.  These are then decoded to
        generate the resulting image."""
        (bs, nf, c, h, w) = O.shape
        X_context = self.x_encoder(O[:, 0:-n, :, :, :])
        if f is None:
            F_mu, F_logvar = self.f_encoder(X_context)
            f = reparameterize(F_mu, F_logvar)
        F_context = f.unsqueeze(1).expand(-1, nf - n, self.f_dim)
        Z_mu, Z_logvar = self.z_encoder(torch.cat((X_context, F_context), dim=-1))
        if Z_context is None:
            Z_context = reparameterize(Z_mu, Z_logvar)
        ZF_context = torch.cat((Z_context, F_context), dim=-1)
        logging.debug("ZF_context.shape={}".format(ZF_context.shape))
        X_next_frames = []
        Z_next_list = []
        for i in range(n):
            if Z_next is None:
                Z_next = self.dynamics_prior(ZF_context, f)[:,-1,:].unsqueeze(1) # Grab just the last predicted frame
            Z_next_list.append(Z_next)
            logging.debug("\n\nZ_next.shape={}".format(Z_next.shape))
            ZF_next = torch.cat((Z_next, f.unsqueeze(1)), dim=-1)
            X_next = self.decoder(ZF_next)
            X_next_frames.append(X_next)
            ZF_context = torch.cat((ZF_context, ZF_next), dim=1)
            logging.debug("sample rollout i={}/{} X_next.shape={}, Z_next.shape={}".format(
                i, n, X_next.shape, Z_next.shape))
        return RolloutResult(X_next=torch.cat(X_next_frames, dim=1),
                             Z_next=torch.cat(Z_next_list, dim=1),
                             f=f)
        
RDSVLoss = collections.namedtuple('RDSVLoss', ['loss', 'elbo', 'kld', 'mse', 'dyn_mse'])

def loss_fn(X_full, r):
    """
    TODO: Add \Beta in front of KL term to implement \Beta VAE ideas
    :param X: Original data sequence to check against reconstruction.  Should be shape (nbatch, frame, c, h, w)
    :param r: DSVResult from DisSeqVAE.forward()
    :return:
    """
    logging.debug("Loss, X.shape={}".format(X_full.shape))
    (bs, nf, nc, h, w) = X_full.shape
    mse = tfunc.mse_loss(X_full, r.X_hat, reduction='sum')
    # KLD of F and Z, presuming "true" prior drawn from N(0,1)
    kld_F = 0.5 * torch.sum(1 + r.F_logvar - torch.pow(r.F_mu, 2) - torch.exp(r.F_logvar))
    kld_Z = 0.5 * torch.sum(1 + r.Z_logvar - torch.pow(r.Z_mu, 2) - torch.exp(r.Z_logvar))
    elbo = mse - (kld_F + kld_Z )
    
    # Dynamics MSE
    dynamics_mse = tfunc.mse_loss(r.Z, r.Z_hat)

    # Return the ELBO (Expected Log-likelihood of Data - KL between variational and prior)
    # TODO Consider the fact that the KL divergence of the last term is equally weighted
    # with the KL divergence of the first term
    return RDSVLoss(loss=elbo + dynamics_mse,
                    elbo=elbo,
                    kld= - (kld_F + kld_Z ),
                    mse=mse,
                    dyn_mse=dynamics_mse)
