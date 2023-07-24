import torch.nn as nn


class GatedConvRegressionHead(nn.Module):
    """
    Implements a sparse numeric value handler, where a sigmoid first determines if a value other than 0 (or the default
    value) should be placed.  Uses 2D spatial input that has a convolution applied to it.
    """
    def __init__(self, c_dim, num_targets=1):
        super(GatedConvRegressionHead, self).__init__()
        self.c_dim = c_dim
        self.num_targets = num_targets
        self.gate = nn.Sequential(nn.ConvTranspose2d(c_dim, 1, kernel_size=1, stride=1),
                                  nn.Sigmoid())
        self.guesser = nn.Sequential(nn.ConvTranspose2d(c_dim, num_targets, kernel_size=1, stride=1),
                                     nn.Sigmoid())
        # Because of the two stage loss, cache the results of the forward, so we can
        # then use it to mask.
        # NOTE: This requires the loss be called immediately after the forward.
        self.last_gate = None
        self.last_guess = None

    def forward(self, X):
        """
        Guesses are performed as a two step process:
        - First perform the sigmoid and construct a mask for > 0.5
        - Apply numeric regressor Hadamard'ed against the mask.
        :param X:
        :return:
        """
        H1 = self.gate(X)
        self.last_gate = H1
        M1 = H1 > 0.5
        H2 = self.guesser(X)
        self.last_guess = H2
        return M1 * H2


class GatedLinearRegressionHead(nn.Module):
    """
    Accepts input from a linear layer, which is then applied.
    """
    def __init__(self, num_in, num_targets=1, add_sigmoid=True):
        super(GatedLinearRegressionHead, self).__init__()
        self.num_in = num_in
        self.num_targets = num_targets
        self.gate = nn.Sequential(nn.Linear(self.num_in, self.num_targets),
                                  nn.Sigmoid())
        if add_sigmoid:
            self.guesser = nn.Sequential(nn.Linear(self.num_in, self.num_targets),
                                         nn.Sigmoid())
        else:
            self.guesser = nn.Sequential(nn.Linear(self.num_in, self.num_targets))
        # Because of the two stage loss, cache the results of the forward, so we can
        # then use it to mask.
        # NOTE: This requires the loss be called immediately after the forward.
        self.last_gate = None
        self.last_guess = None

    def forward(self, X):
        """
        Guesses are performed as a two step process:
        - First perform the sigmoid and construct a mask for > 0.5
        - Apply numeric regressor Hadamard'ed against the mask.
        :param X:
        :return:
        """
        H1 = self.gate(X)
        self.last_gate = H1
        M1 = H1 > 0.5
        H2 = self.guesser(X)
        self.last_guess = H2
        return M1 * H2