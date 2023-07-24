import os
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset
from imago_prev.data.datasets import load_metadata, XTypeEnum, get_obs_tensors

class TensorTrajectoryDataset(Dataset):
    """
    Base dataset for treating each trajectory as an instance. This loads the npz files
    from npz_dir, saved out by RawTrajectoryDataset.save().
    If the filter function is active, it is executed accepts the (Xs, Ys, and Cs) for the
    given trajectory.  If True, the trajectory is accepted, otherwise it's excluded.

    See imago.data.save_replays_as_tensors for generating these NPZs from the replay files.
    """
    def __init__(self, npz_dir, metadata=None, tpool_size=20, filter_fn=None, trajectories=None):
        if npz_dir is None and trajectories is not None:
            self.metadata = metadata
            self.trajectories = trajectories
        else:
            if metadata is None:
                self.metadata = load_metadata(os.path.join(npz_dir, "metadata.json"))
            else:
                self.metadata = metadata
            self.trajectories = []
            npz_files = [os.path.join(npz_dir, fname) for fname in os.listdir(npz_dir) \
                                        if fname.endswith(".npz")]
            self.trajectories = []
            self.filter_fn = filter_fn

            def _load(npz_file):
                npz = np.load(npz_file)
                Xs, Ys, Cs = npz['Xs'], npz['Ys'], npz['Cs']
                if not(filter_fn):
                    return (Xs, Ys, Cs)
                else:
                    if filter_fn(Xs, Ys, Cs):
                        return (Xs, Ys, Cs)
                    else:
                        return None
            with ThreadPool(tpool_size) as p:
                self.trajectories = list(tqdm(p.imap(_load, npz_files), total=len(npz_files)))
                self.trajectories = [x for x in self.trajectories if x is not None]

    def __getitem__(self, item):
        """
        Returns the trajectory as a single batch tensor length # of observations
        :param item:
        :return:
        """
        return self.trajectories[item]

    def __len__(self):
        return len(self.trajectories)
    
    def filter(self, filter_fn):
        new_trajectories = [x for x in self.trajectories if filter_fn(*x)]
        return TensorTrajectoryDataset(None, metadata=self.metadata, trajectories=new_trajectories)

class RawTrajectoryDataset(Dataset):
    """
    Base dataset for treating each trajectory as an instance
    """
    def __init__(self, observations, metadata, add_obs=False, x_type=XTypeEnum.UNIT_TYPE_ONEHOT):
        self.x_type = x_type
        self.metadata = metadata
        self.trajectories = []
        curr_trajectory = []
        for obs_group in observations:
            for obs in obs_group:
                X, _, _ = get_obs_tensors(obs, self.x_type)
                if np.sum(X) == 0:
                    # Blank screen, begin recording as a new trajectory
                    if len(curr_trajectory) > 0:
                        self.trajectories.append(curr_trajectory)
                    curr_trajectory = []
                else:
                    curr_trajectory.append(obs)
        if len(curr_trajectory) > 0:
            self.trajectories.append(curr_trajectory)

    def __getitem__(self, item):
        """
        Returns the trajectory as a single batch tensor length # of observations
        :param item:
        :return:
        """
        raw_trajectory = self.trajectories[item]
        Xs, Ys, Cs = [], [], []
        for obs in raw_trajectory:
            X, Y , C = get_obs_tensors(obs, self.x_type)
            Xs.append(X)
            Ys.append(Y)
            Cs.append(C)
        Xs = np.array(Xs)
        Ys = np.array(Ys)
        Cs = np.array(Cs)
        return Xs, Ys, Cs

    def __len__(self):
        return len(self.trajectories)

    def save(self, tgtdir_fpath, start_idx=0):
        os.makedirs(tgtdir_fpath, exist_ok=True)
        for t_idx in tqdm(range(len(self))):
            tgt_fpath = os.path.join(tgtdir_fpath, "t{}".format(t_idx + start_idx))
            Xs, Ys, Cs = self[t_idx]
            np.savez(tgt_fpath, Xs=Xs, Ys=Ys, Cs=Cs)