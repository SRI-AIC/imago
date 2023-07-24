from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from imago import losses
from imago.layers import GatedLinearRegressionHead
from imago.spaces import ShapedChannelSpec
from imago.spaces.channels import SPACE_TYPE
import pdb

class MultiMLPOutput(nn.Module):
    """
    Constructs multiple MLP outputs given a space specification
    """
    def __init__(self, space_specifier,
                 device="cpu"):
        super(MultiMLPOutput, self).__init__()
        self.space_specifier=space_specifier
        self.device = device

class MLPOutput(nn.Module):
    """
    Goes from the latent of the model, through a MLP, to the target
    """
    def __init__(self, space_specifier: ShapedChannelSpec,
                 latent_size,
                 device="cpu",
                 use_block2=False):  # block2 added to try to use tanh, but currently no effect
        super(MLPOutput, self).__init__()
        self.space_specifier = space_specifier
        self.device = device
        self.latent_size = latent_size

        # for this, perhaps just focus on numerics, categoricals will need
        # a channel of their own
        if space_specifier.space_type == SPACE_TYPE.CATEGORICAL:
            if isinstance(self.space_specifier.shape, int):
                num_targets = self.space_specifier.shape * self.space_specifier.size
            elif len(self.space_specifier.shape) == 1:
                num_targets = self.space_specifier.shape[0] * self.space_specifier.size
            else:
                num_targets = reduce(lambda x, y: x * y, self.space_specifier.shape) * self.space_specifier.size
        else:
            if isinstance(self.space_specifier.shape, int):
                num_targets = self.space_specifier.shape
            elif len(self.space_specifier.shape) == 1:
                num_targets = self.space_specifier.shape[0]
            else:
                num_targets = reduce(lambda x, y: x*y, self.space_specifier.shape)

        # Assign sigmoid activation only if we are targeting a categorical or
        # distributional target.  Otherwise leave as is for regression.
        if use_block2:
            self.network = _mlp_block2(self.latent_size, num_targets).to(device)
        else:
            add_sigmoid = space_specifier.space_type == SPACE_TYPE.CATEGORICAL or \
                space_specifier.is_distributional
            self.network = _mlp_block(self.latent_size, num_targets,
                                      add_sigmoid=add_sigmoid).to(device)

    def forward(self, X):
        # Unpack the output to match
        if isinstance(self.space_specifier.shape, int):
            return self.network(X).view(-1, self.space_specifier.shape)
        return self.network(X).view(-1, *self.space_specifier.shape)

    def loss(self, Yhat: torch.Tensor, Y: torch.Tensor):
        """ Given the guess produced by forward, the target tensor passed in,
        computes the loss.

        Note we have to explode a Box into target tensors by constituent channels,
        as they may have entirely different objectives.
        """
        loss = 0.
        batch_size = Y.shape[0]
        space_type = self.space_specifier.space_type

        if space_type == SPACE_TYPE.CATEGORICAL:
            # Reshape into a (batch, *space_shape, num_targets) shape
            Yhat = Yhat.view(-1, *self.space_specifier.shape, self.space_specifier.size)
            hyp = F.log_softmax(Yhat, dim=-1)
            loss += F.nll_loss(hyp, Y.to(dtype=torch.long,
                                         device=self.device))  # Remove last so we get H,W,target
        elif space_type == SPACE_TYPE.NUMERIC:
            Y = Y.to(dtype=torch.float, device=self.device)
            if len(Y.shape) == 1:
                Y = Y.view(batch_size, 1) # Expand out so MSE Loss works
            Y = self.space_specifier.norm(Y) # Move target into norm'ed space
            # Squeezing the target, as the target has the extra dimension for the value
            loss += F.mse_loss(Yhat, Y)
        elif space_type == SPACE_TYPE.SPARSE_NUMERIC:
            # Do a two stage check.  First treat it as a 0-1 problem, then
            # regress the value just on the mask of guessed values.
            H1 = self.network[-1].last_gate
            T1 = Y.to(self.device, dtype=torch.float)
            T1[torch.where(Y > 0)] = 1
            loss1 = F.binary_cross_entropy(H1, T1)
            loss += loss1

            # 2nd stage, mask against 0/1 targets in stage 1 and regress
            # Note we mask against the gold mask, as other values are zeroed and we
            # do not want to sparsify against any false positives on the mask.
            H2 = H1 * T1
            T2 = Y.to(dtype=torch.float, device=self.device)
            T2 = self.space_specifier.norm(T2)
            loss2 = F.mse_loss(H2, T2)
            loss += loss2
        elif self.space_specifier.is_distributional:
            Y = Y.to(dtype=torch.float, device=self.device)
            Y = self.space_specifier.norm(Y)
            Y = Y.view(batch_size, *self.space_specifier.shape)
            if space_type == SPACE_TYPE.DIST_JSD:
                divg_fn = losses.JSD
            elif space_type == SPACE_TYPE.DIST_KL_DIV:
                divg_fn = losses.KLDIV
            else:
                raise Exception("Unsupported distribution space type={}".format(space_type))
            loss += losses.dist_divg_fn(Yhat, Y, divg_fn)
        else:
            raise Exception("Unsupported channel type={}".format(self.space_specifier.space_type))
        return loss


def _mlp_block(num_in, num_out, add_sigmoid=False, dropout_rate=0.):
    """
    Sets up a MLP for translating the VAE encoder into the target representation.
    Uses a winnowing architecture, going from a larger representation to a smaller target
    """
    fc1_size = (max(num_in, num_out) + 1) // 2
    fc2_size = (max(num_in, num_out) + 1) // 4
    fc3_size = (max(num_in, num_out) + 1) // 8
    base_layers = [
        nn.Linear(num_in, fc1_size),
        nn.Dropout(p=dropout_rate),
        nn.ReLU(),
        nn.Linear(fc1_size, fc2_size),
        nn.Dropout(p=dropout_rate),
        nn.ReLU(),
        nn.Linear(fc2_size, fc3_size),
        nn.Dropout(p=dropout_rate),
        nn.ReLU(),
        nn.Linear(fc3_size, num_out),
    ]
    if add_sigmoid:
        base_layers.append(nn.Sigmoid())
    return nn.Sequential(
        *base_layers
    )

def _mlp_block2(num_in, num_out, dropout_rate=0.):
    """
    Sets up a MLP for translating the VAE encoder into the target representation
    """
    fc1_size = (max(num_in, num_out) + 1) // 2
    fc2_size = (max(num_in, num_out) + 1) // 4
    fc3_size = (max(num_in, num_out) + 1) // 8
    base_layers = [
        nn.Linear(num_in, fc1_size),
        nn.Dropout(p=dropout_rate),
        nn.ReLU(),
        nn.Linear(fc1_size, fc2_size),
        nn.Dropout(p=dropout_rate),
        nn.ReLU(),
        nn.Linear(fc2_size, fc3_size),
        nn.Dropout(p=dropout_rate),
        nn.ReLU(),
        nn.Linear(fc3_size, num_out)
        #nn.Tanh()
    ]
    return nn.Sequential(
        *base_layers
    )

def _gated_regression_block(num_in, num_out, dropout_rate=0.):
    """
    Sets up a MLP for translating the VAE encoder into the target representation
    """
    fc1_size = (max(num_in, num_out) + 1) // 2
    fc2_size = (max(num_in, num_out) + 1) // 4
    return nn.Sequential(
        nn.Linear(num_in, fc1_size),
        nn.Dropout(p=dropout_rate),
        nn.ReLU(),
        nn.Linear(fc1_size, fc2_size),
        nn.Dropout(p=dropout_rate),
        nn.ReLU(),
        GatedLinearRegressionHead(fc2_size, num_targets=num_out)
    )
