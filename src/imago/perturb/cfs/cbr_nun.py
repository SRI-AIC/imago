"""
Case-based counterfactual development using nearest unlike neighbors (NUNs)
"""
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from dataclasses import dataclass

from imago.perturb.cfs import CounterFactual
from imago.domains import OBS, ImagoDomain, forward
from imago.analysis.walk import ZTrajectory
from imago.utils import ensure_numpy

IDX = "idx"
DATUM = "datum"
ODIFF = "ODiff"

INST_DS = 'inst_ds'

NUN = "NUN"
INTRP_PT1 = "InterpPt1"
INTRP_PT2 = "InterpPt2"


def get_cfs(query_datum, perturb, domain, inst_ds, **kwargs):
    """ Computes the counterfactuals """
    cbr_nun = CBR_NUN(query_datum, perturb, inst_ds, domain, **kwargs)
    cfs = { }
    nun_datum = inst_ds[cbr_nun.neighborhood[0].idx]
    _, _, _, nun_Z, _ = forward(domain.model, nun_datum)
    cfs[NUN] = CounterFactual(query_datum=query_datum,
                              cf_datum = nun_datum,
                              cf_Z = nun_Z,
                              odiff = cbr_nun.neighborhood[0].odiff)
    # Get the ZTrajectory for the NUN and get the pt1 and pt2
    ztraj = cbr_nun.get_ztraj(0)
    pt1, pt2, _ = ztraj.compute_CF_points(perturb)
    #if pt1 is not None:
    #    Ohat_pt1, That_pt1 = domain.model.forward_Z(pt1.)

import pdb

@dataclass
class Neighbor:
    idx: int
    datum:dict
    odiff:float

class CBR_NUN:
    """
    Nearest unlike neighbor (NUN) counterfactual identification method.  Given a query, directions of
    perturbation, the dataset of instances to examine, and the domain (model, odiff function, render
    function), constructs the neighborhood .
    """
    def __init__(self, query_datum, perturb, inst_ds:Dataset,
                 domain:ImagoDomain, search_stepsize=5,
                 stop_at_first=False):
        self.query_datum = query_datum
        self.perturb = perturb
        self.inst_ds = inst_ds
        self.domain = domain
        self.search_stepsize=search_stepsize

        # Generate encodings for start point
        _, _, _, Z, _ = forward(self.domain.model, query_datum)
        self.query_Z = ensure_numpy(Z)

        """ Gets the indices of neighbors to this observation, based
        upon observational difference. """
        start_value = query_datum[self.perturb.var_name]
        nn_tuples = []
        for idx in range(0, len(self.inst_ds), self.search_stepsize):
            datum = self.inst_ds[idx]
            Op = datum[OBS]
            curr_value = datum[self.perturb.var_name]
            cf_target_met = self.perturb.target_met(start_value, curr_value)
            if cf_target_met:
                odiff = self.domain.odiff_fn(np.expand_dims(query_datum[OBS], 0), np.expand_dims(Op, 0))
                if odiff > 0:
                    # Avoid capturing exact matches
                    nn_tuples.append(Neighbor(idx=idx, datum=datum, odiff=odiff))
                if stop_at_first:
                    break
        self.neighborhood = sorted(nn_tuples, key=lambda x: x.odiff)

    def __str__(self):
        return "CBR NUNs: # Valid Neighbors={}, var={}".format(len(self.neighborhood), self.perturb.var_name)

    def get_neighbor(self, idx):
        if idx >= len(self.neighborhood):
            return None
        return self.neighborhood[idx]

    def get_ztraj(self, neighbor_idx):
        """ Constructs the variable trajectories for going from the given query point to
        the desired neighbor"""
        if neighbor_idx >= len(self.neighborhood):
            return None
        inst_idx = self.neighborhood[neighbor_idx].idx
        n_datum = self.inst_ds[inst_idx]
        _, _, _, end_Z, _ = forward(self.domain.model, n_datum)
        return ZTrajectory(self.query_Z, ensure_numpy(end_Z), self.domain)