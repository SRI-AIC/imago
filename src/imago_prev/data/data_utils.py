import numpy
from tqdm import tqdm
from sc2recorder.utils import ensure_numpy

def compare_datasets(ds1, ds2):
    """ Given two datasets, runs a comparison to flag which instances are similar between them.
    Run a simple hash using the first element returned by getitem() on each dataset, presuming 
    that's the data matrix.
    
    Returns a list of integer pairs indicating matches, (idx in dataset1, idx in dataset 2)
    """
    seen_ds1 = set()
    collision_idxes = []
    seen_ds1_idx = {}
    for idx in tqdm(range(len(ds1))):
        mats = ds1[idx]
        x1 = ensure_numpy(mats[0])
        k = x1.tostring()
        seen_ds1.add(k)
        seen_ds1_idx[k] = idx
    for idx in tqdm(range(len(ds2))):
        mats = ds2[idx]
        x2 = ensure_numpy(mats[0])
        k = x2.tostring()
        if k in seen_ds1:
            collision_idxes.append((seen_ds1_idx[k], idx))
    return collision_idxes