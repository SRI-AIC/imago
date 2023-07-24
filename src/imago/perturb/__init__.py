import pdb
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import torch

from imago.utils import ensure_torch, ensure_numpy

@dataclass
class Perturb:
    var_name:str               # Identifier for target variable to perturb
    direction:float=None  # Desired direction +/- or None (any)
    scale:float=1.0       # Scale to apply for sampling search
    tgt_mag:float=0.5     # Target difference in magnitude, for counterfactuals

    def __str__(self):
        return "{}d={}s={:.3f}".format(self.var_name, self.direction, self.scale)

    def get_target_value(self, start_value):
        if self.direction > 0:
            return (start_value + self.tgt_mag)
        elif self.direction < 0:
            return (start_value - self.tgt_mag)
        raise Exception("Unsupported direction value={}".format(self.direction))

    def target_met(self, start_value, cand_value):
        """ See if the candidate value matches the target value, which is composed
        of the starting and target magnitude of change."""
        if self.direction > 0:
            return cand_value >= self.get_target_value(start_value)
        elif self.direction < 0:
            return cand_value <= self.get_target_value(start_value)
        raise Exception("Unsupported direction value={}".format(self.direction))

    def get_good_queries(self, ds, rng_seed=1337,
                    limit=1000,
                    search_magnitude=2,
                    var_max_mag=1.0):
        """ Given a dataset of observations, identifies those that can act best
        as queries for demonstrating this perturbation.  These correspond to
        instances that have a variable value opposite of the direction the
        perturbation moves in."""
        perturb_var = self.var_name

        # Get all of the instances, sort them in order
        sorted_idxes = sorted([(idx, datum[perturb_var][0]) for idx, datum in enumerate(ds)],
                              key=lambda x: x[1])
        if self.direction < 0:
            sorted_idxes.reverse()

        # Create candidates from the extrema
        if self.direction > 0:
            upper_bound = sorted_idxes[-1][1]
            candidates = [pair for pair in sorted_idxes[0:(limit * search_magnitude)]
                          if pair[1] < upper_bound]
        elif self.direction < 0:
            lower_bound = sorted_idxes[-1][1]
            candidates = [pair for pair in sorted_idxes[0:(limit * search_magnitude)]
                          if pair[1] > lower_bound]
        # Randomize the filtered candidates and take the first LIMIT as the schedule
        rng = np.random.default_rng(rng_seed)
        rng.shuffle(candidates)
        schedule = candidates[0:limit]
        return [pair[0] for pair in schedule]


@dataclass
class DeltaVar:
    var:str
    dv:np.array
    def stats(self):
        return np.mean(self.dv), np.std(self.dv), np.min(self.dv), np.max(self.dv), np.sum(self.dv)

    def stats_str(self):
        mu, std, mi, ma, su = self.stats()
        return "{}\tmean/std={:.5f}/{:.5f}, [{:.5f}, {:.5   f}] sum={:.5f}".format(self.var,
                                                                               mu, std, mi, ma, su)


@dataclass
class AnalysisPoint:
    Odiff: float
    deltas: dict
    perturb: Perturb

    def set_inflection_step(self, fraction, direction):
        self.fraction = fraction
        self.direction = direction

    def get_dir_fraction(self):
        if hasattr(self, "step"):
            return self.fraction
        return None

    def has_direction(self):
        return hasattr(self, "direction") and self.direction is not None

    def get_direction(self):
        if hasattr(self, "direction"):
            return self.direction
        return None

    def __str__(self):
        ret = "Odiff={:.5f}".format(self.Odiff)
        if self.get_dir_fraction() is not None:
            ret += " Frac={:.3f}".format(self.fraction)
        ret += "\n"
        for var_name, var_delta in self.deltas.items():
            if var_name == self.perturb.var_name:
                line = "* d {}:\t{:.5f}\n".format(var_name, var_delta)
            else:
                line = "d {}:\t{:.5f}\n".format(var_name, var_delta)
            ret += line
        return ret

    def is_valid(self):
        """ Check to see if the delta in the variable matches
        the criteria in the perturbation"""
        diff = self.deltas[self.perturb.var_name]
        if self.perturb.direction < 0:
            return diff < -1 * abs(self.perturb.scale)
        elif self.perturb.direction > 0:
            return diff > abs(self.perturb.scale)
        else:
            raise Exception("Unsupported perturb direction={}".format(self.perturb.direction))


class PointAnalyzer:
    def __init__(self, model, perturb, start_Z):
        self.device = model.device
        self.model, self.perturb, self.start_Z = model, perturb, start_Z
        start_Ohat, self.start_That = model.forward_Z(ensure_torch(self.device, start_Z))
        self.start_Ohat = ensure_numpy(start_Ohat)

    def analyze(self, Z):
        Oh, Th = self.model.forward_Z(ensure_torch(self.device, Z))
        Odiff = self.model.space_specifier.compute_Odiff(Oh, self.start_Ohat)
        #Oh = ensure_numpy(Oh)
        #Odiff = np.sum(np.abs(Oh - self.start_Ohat))
        deltas = { var_name: np.sum(ensure_numpy(var_value - self.start_That[var_name]))\
                   for var_name, var_value in Th.items()}
        ap = AnalysisPoint(Odiff=Odiff, deltas=deltas, perturb=self.perturb)
        ap.set_inflection_step(1.0, Z)  # Set to full direction first
        return ap

    def scaled_analysis(self, Z, min_Odiff=1, norm_dir=True, num_ticks=1000, verbose=True):
        """
        Given a direction, walks increments until the minimum Odiff is
        encountered.  If none, returns None.
        :param Z:
        :return:
        """
        direction = ensure_torch(self.device, Z) - ensure_torch(self.device, self.start_Z)
        if norm_dir:
            direction /= torch.linalg.norm(direction)
        if verbose:
            iter_obj = tqdm(range(1, num_ticks + 1))
        else:
            iter_obj = range(1, num_ticks + 1)
        for i in iter_obj:
            probe_Z = ensure_torch(self.device, self.start_Z) + i / num_ticks * direction
            ap = self.analyze(probe_Z)
            if verbose:
                iter_obj.set_description("Odiff={:.5f}".format(ap.Odiff))
            if ap.Odiff >= min_Odiff and ap.is_valid():
                ap.set_inflection_step(i / num_ticks, probe_Z)
                return ap
        return None