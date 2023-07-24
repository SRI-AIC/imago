from pathlib import Path
from skimage import measure as ski_measure
import numpy as np
import pandas as pd
from tqdm import tqdm
import platform
from multiprocessing.pool import Pool
import time

from imago.io_utils import load_object, save_object
from imago.domains import DictTensorDataset, load_from_npz
from imago.domains import summary_str, var_shortname, VALUE_FUNCTION, CONFIDENCE, GOAL_COND, RISKINESS, INCONGRUITY
from imago import PVAE_ROOT
from imago.utils import flatten

N_WORKERS=4

CAML_ROOT = Path(PVAE_ROOT, "..", "..")
DATA_ROOT = Path(CAML_ROOT, "datasets", "ReaverAssaultYear2")

DEFAULT_DATA_FPATH = Path(DATA_ROOT, "interaction_data.pkl.gz")
DEFAULT_CSV_FPATH = Path(DATA_ROOT, "interestingness.csv.gz")
DEFAULT_ROLLOUTS_DIR = Path(DATA_ROOT, "strided_rollouts")

def downsample_frame(frame):
    """ Downsample from Bx3x144x192 to Bx3x72,64, then take center 64 rows"""
    F2 = ski_measure.block_reduce(frame, (1, 1, 2, 3), func=np.max)
    F2a = F2[:, :, 4:68, :]  # Note: Strips away top 4 and bottom 4 rows
    return F2a


def _load_rollouts_from_dir(rollouts_dir=DEFAULT_ROLLOUTS_DIR,
                            num_workers=N_WORKERS,
                            load_debug=False,
                            use_mp=True):
    pkl_files = sorted(list(Path(rollouts_dir).glob("*.pkl.gz")))
    if load_debug:
        pkl_files = pkl_files[0:10]
    start_time = time.perf_counter()
    if not(use_mp):
        print("Force using single process")
        rollouts = []
        for pkl_file in tqdm(pkl_files, total=len(pkl_files)):
            rollouts.append(load_object(pkl_file))
        end_time = time.perf_counter()
        print("Loading from dir total time={}".format(end_time - start_time))
        return {ep.rollout_id: ep for ep in rollouts}
    else:
        with Pool(num_workers) as p:
            rollouts = list(tqdm(p.imap_unordered(load_object, pkl_files), total=len(pkl_files)))
            end_time = time.perf_counter()
            print("Loading from dir total time={}".format(end_time - start_time))
            return { ep.rollout_id: ep for ep in rollouts }


def _save_fn(triplet):
    (rollouts_dir, name, ep) = triplet
    save_object(ep, Path(rollouts_dir, "{}.pkl.gz".format(name)))


def _save_rollouts2dir(rollouts, rollouts_dir=DEFAULT_ROLLOUTS_DIR, num_workers=N_WORKERS):
    Path(rollouts_dir).mkdir(exist_ok=True, parents=True)
    with Pool(num_workers) as p:
        schedule = [(rollouts_dir, name, ep) for name, ep in rollouts.items()]
        list(tqdm(p.imap_unordered(_save_fn, schedule), total=len(rollouts)))


def _ep2np(ep):
    O = ep.data.observation.astype(np.float64)  # (Frames, ... )
    O = downsample_frame(O)
    R = ep.data.reward
    V = ep.data.value
    return ep.rollout_id, O, R, V



def load_data(data_fpath=DEFAULT_DATA_FPATH, intr_csv_fpath=DEFAULT_CSV_FPATH,
              tt_splitat=0.95, load_debug=False,
              rollouts_dir=DEFAULT_ROLLOUTS_DIR,
              use_mp=True):
    if platform.system() != "Linux":
        print("Currently on non-Linux system, setting multiprocess loading of data to False for stability issues.")
        use_mp = False
    rollouts_dir = Path(rollouts_dir)
    if rollouts_dir.exists():
        # Load from mp
        print("Rollout strided dir exists, loading from {}".format(rollouts_dir))
        rollouts = _load_rollouts_from_dir(rollouts_dir=DEFAULT_ROLLOUTS_DIR, load_debug=load_debug,
                                           use_mp=use_mp)
    else:
        print("No rollouts dir, loading full pkl")
        start_time = time.perf_counter()
        rollouts = load_object(data_fpath)
        end_time = time.perf_counter()
        print("Loading total time={}".format(end_time - start_time))
        print("Saving strided cache to {}".format(rollouts_dir))
        _save_rollouts2dir(rollouts, rollouts_dir=rollouts_dir)  # Cache strided for easier loading

    tt_rollout_index = int(len(rollouts) * tt_splitat)
    tt_eidx = 0
    assert tt_rollout_index > 0
    vars_df = pd.read_csv(intr_csv_fpath)  # Interestingness variables dataframe
    Os = []
    targets = {
                VALUE_FUNCTION: [],
                CONFIDENCE: [],
                GOAL_COND: [],
                RISKINESS: [],
                INCONGRUITY: []
            }
    entries = []
    if not(use_mp):
        for rv in tqdm(rollouts.values(), total=len(rollouts)):
            entries.append(_ep2np(rv))
    else:
        with Pool(N_WORKERS) as p:
            entries = list(tqdm(p.imap_unordered(_ep2np, rollouts.values()), total=len(rollouts)))
    entries = sorted([(name, (O, R, V)) for name, O, R, V in entries],
                     key=lambda x: x[0])
    entries = dict(entries)

    ep_ids = []
    frame_ids = []
    print("Unboxing rollout and interaction data")
    for rollout_num, t in tqdm(enumerate(entries.items()), total=len(entries)):
        """
        Unbox the Rollout and Interaction Data
        """
        rid, (O, R, V) = t
        intr_df = vars_df[vars_df.Rollout == rid]  # TODO: Assert these exist
        assert len(intr_df) == len(O) == len(R) == len(V)

        #TODO:Resacle to 64x64 from 3x144x192

        # Unroll the episode into individual instances
        Os.append(O)
        targets[VALUE_FUNCTION].append(V)
        for colname in [CONFIDENCE, GOAL_COND, RISKINESS, INCONGRUITY]:
            arr = intr_df[colname].to_numpy().reshape((-1, 1))  # Padd with a column value so vstack works
            targets[colname].append(arr)
        # Identify episode split point
        if rollout_num < tt_rollout_index:
            tt_eidx += len(O)
        ep_ids.append([rid for _ in range(len(O))])
        frame_ids.append(list(range(0, len(O))))
    assert tt_eidx > 0
    Os_train, Os_test = np.vstack(Os[0:tt_rollout_index]), np.vstack(Os[tt_rollout_index:])
    targets_train, targets_test = {}, {}
    for k, v in targets.items():
        targets_train[k] = np.vstack(v[0:tt_rollout_index]).astype(np.float64)
        targets_test[k] = np.vstack(v[tt_rollout_index:]).astype(np.float64)
    epfr_df_train = pd.DataFrame({
        "episode": flatten(ep_ids[0:tt_rollout_index]),
        "frame": flatten(frame_ids[0:tt_rollout_index])
    })
    epfr_df_test = pd.DataFrame({
            "episode": flatten(ep_ids[tt_rollout_index:]),
            "frame": flatten(frame_ids[tt_rollout_index:])
        })
    train_dataset = DictTensorDataset(Os_train, targets_train, epfr_lookup=epfr_df_train)
    test_dataset = DictTensorDataset(Os_test, targets_test, epfr_lookup=epfr_df_test)
    return train_dataset, test_dataset


