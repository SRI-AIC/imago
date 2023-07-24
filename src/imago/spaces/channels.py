from enum import IntEnum

import numpy as np

DEFAULT_EMBEDDING_DIM = 10


class SPACE_TYPE(IntEnum):
    NUMERIC = 0  # Target using MSE
    SPARSE_NUMERIC = 1 # Numeric, but most values are 0 (uses GatedRegressionHeads)
    STANDARDIZED = 2 # Standardized numeric value, centered at 0. DEPRECATED, replaced with numerics that are centered
    CATEGORICAL = 3  # Create embedding for each category type
    DIST_KL_DIV = 4  # Distribution, use KL Divergence to tune
    DIST_JSD = 5  # Distribution, use Jensen-Shannon Divergence to tune
    BINARY = 6  # Binary array, just one or none




class ChannelSpec():
    """ Speciication for a channel in a Box space. Low and high values can be
    specified, or inferred from the Box space itself.

    By default the values are normed to 0-1 range.  If apply_norm is set to
    False, this is not performed.

    TODO: Subclass one with shape
    """
    def __init__(self, name, space_type,
                 shape=None,
                 low=None, high=None,
                 mean=None, std=None,
                 channel_idx=None,  # If this is in a multichannel space, this is the channel offset
                 apply_whitening=False,  # Apply normalization, to high-low range
                 apply_recenter=False,  # If True, uses mean and std
                 ):
        self.name, self.space_type = name, space_type
        self.shape = shape
        self.low, self.high = low, high
        self.mean, self.std = mean, std
        self.channel_idx = channel_idx
        self.apply_whitening = apply_whitening
        self.apply_recenter = apply_recenter
        if self.is_distributional:
            self.low = 0
            self.high = 1

    def __str__(self):
        return "CSpec {}: {} cidx={}".format(self.name, self.space_type, self.channel_idx)

    @property
    def size(self):
        # If this is a categorical type, we have to offset by 1 to account for
        # zero indexing.  Otherwise use regular range.
        if self.space_type == SPACE_TYPE.CATEGORICAL:
            return 1 + self.high - self.low
        elif self.space_type == SPACE_TYPE.BINARY:
            return 1
        else:
            return self.high - self.low

    @property
    def is_categorical(self):
        return self.space_type == SPACE_TYPE.CATEGORICAL

    @property
    def is_numeric(self):
        return self.space_type == SPACE_TYPE.NUMERIC or self.space_type == SPACE_TYPE.SPARSE_NUMERIC

    @property
    def is_distributional(self):
        return self.space_type == SPACE_TYPE.DIST_JSD or self.space_type == SPACE_TYPE.DIST_KL_DIV

    def norm(self, A):
        """ Transform the data into the desired normalized format.
        NOTE: Operators are in-place, original tensor will be modified!"""
        if self.apply_whitening:
            A = self._whiten(A)
        if self.apply_recenter:
            A = self._center(A)
        return A

    def denorm(self, A):
        """ Reverse any transforms to return data back in original space.
        NOTE: Operators are in-place, original tensor will be modified!"""
        if self.apply_whitening:
            A = self._dewhiten(A)
        if self.apply_recenter:
            A = self._decenter(A)
        return A

    def _whiten(self, A):
        """ Data 'whitening': Given a space and an input A, norms according to the maximum value and shifts so
        the space range is between 0-1."""
        space_high = np.max(self.high)
        space_low = np.min(self.low)
        assert space_high > space_low
        range_size = space_high - space_low
        A = (A - space_low) / range_size
        return A

    def _dewhiten(self, X):
        """ Given a space and an input X, returns it back to the low-high range given by the space specification."""
        space_high = np.max(self.high)
        space_low = np.min(self.low)
        assert space_high > space_low
        range_size = space_high - space_low
        X = (X * range_size) + space_low
        return X

    def _center(self, A):
        return (A - self.mean) / self.std

    def _decenter(self, A):
        return (A * self.std) + self.mean
