"""
Implements a two step reconstruction anomaly score
based off of observational differences
"""

from imago.utils import ensure_torch

class TwoStepODiffAnomDet:
    def __init__(self, model,
                 odiff_fn,
                 odiff_thresh=107):
        self.model, self.odiff_fn = model, odiff_fn
        self.device = model.device
        self.odiff_thresh=odiff_thresh

    def forward(self, Z):
        Z = ensure_torch(self.device, Z)
        Ohat, That = self.model.forward_Z(Z, ret_for_training = False)
        Ohat2, _, _, Z2, That2 = self.model(Ohat)
        Odiff = self.odiff_fn(Ohat, Ohat2)
        return Ohat, That, Ohat2, Z2, That2, Odiff

    def score(self, Z):
        Ohat, That, Ohat2, Z2, That2, Odiff = self.forward(Z)
        label = 0
        if Odiff >= self.odiff_thresh:
            label = 1
        return Odiff, label

