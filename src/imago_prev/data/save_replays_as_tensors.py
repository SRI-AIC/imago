"""
Routine used to convert replays into tensortrajectory data, saved out to tensor_replays.
Use this to save significant amounts of time with unboxing the JSON
"""

from data.datasets import *
from sc2recorder.record_utils import *
from imago_prev.models.supervised.supervised_condvaes import RawTrajectoryDataset
from sc2recorder.utils import set_random_seed
from absl import logging

ROOT_DIR="."
train_dir = "{}/data/featurized_replays".format(ROOT_DIR)
SAVE_DIR = "{}/data/tensor_replays".format(ROOT_DIR)

set_random_seed()
os.makedirs(SAVE_DIR, exist_ok=True)

def identify_episodes(root_dir, debug=False):
    files = []
    for fname in os.listdir(root_dir):
        m = re.search('.*(\\d+).json.gz', fname)
        if m is None:
            # May not be integer keyed, use prefix instead
            m2 = re.search('(.+).json.gz', fname)
            if m2 is not None:
                files.append((fname, m2.group(1)))
            else:
                logging.warning("Invalid json.gz record name={}".format(fname))
        else:
            files.append((fname, int(m.group(1))))
    if debug:
        # take every 5 files instead
        debug_files = []
        for idx, file in enumerate(files):
            if idx % 253 == 0:
                debug_files.append(file)
        logging.info("Debug, using smaller set size={}, orig size={}".format(len(debug_files),
                                                                             len(files)))
        files=debug_files # Truncate for debug version
    files.sort(key=lambda x: x[1])
    files = [x[0] for x in files]
    files = [os.path.join(root_dir, f) for f in files]
    return files


def process_files(files):
    with ThreadPool(10) as p:
        # Attempt to load each episode.  Remove failures (value None).
        episodes = p.map(load_episode_json, files)
        episodes = [x for x in episodes if x is not None]
    print("Process, len episodes={}".format(len(episodes)))
    metadata = featurize_simple_combat1_eps(episodes, fnames=[os.path.basename(f) for f in files])
    metadata.save(os.path.join(SAVE_DIR, "metadata.json"))
    ttd = RawTrajectoryDataset(episodes, metadata)
    total_in_dir = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".npz")])
    ttd.save(SAVE_DIR, start_idx=total_in_dir)

stride_N=10
all_eps = identify_episodes(train_dir)
for idx in tqdm(range(0, len(all_eps), stride_N)):
    print("Next stride idx={}/{}".format(idx, len(all_eps)))
    files = all_eps[idx:idx+stride_N]
    process_files(files)

