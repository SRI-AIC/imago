"""
Sets up Canniballs data using the new Interestingness Variables

"""
import pandas as pd
import numpy as np
import imago.viz
from gym.spaces.box import Box

from imago import *
from imago.io_utils import load_object
from pathlib import Path
from imago.domains import DictTensorDataset
from imago.models.model import PVaeModel
from imago.model_utils import load_checkpoint
from imago.spaces import SPACE_TYPE, MCBoxSpec
from imago.spaces.channels import ChannelSpec
from tqdm import tqdm

OBSERVATION = 'observation'
ACTION_DIST = 'action_dist'
ACTION_LOGITS = 'action_logits'
VALUE_FUNCTION = 'value_function'

CONFIDENCE = "Confidence"
GOAL_COND = "Goal Conduciveness"
RISKINESS = "Riskiness"
INCONGRUITY = "Incongruity"

if True:
    CHANNEL_SPECS = [
        ChannelSpec("agent_pos", SPACE_TYPE.CATEGORICAL, low=0, high=1),
        ChannelSpec("opponents", SPACE_TYPE.CATEGORICAL, low=0, high=1),
        ChannelSpec("food", SPACE_TYPE.CATEGORICAL, low=0, high=1),
        ChannelSpec("obstacles", SPACE_TYPE.CATEGORICAL, low=0, high=1)
    ]

#ChannelSpec("agent_pos", SPACE_TYPE.CATEGORICAL, low=0, high=1),
#ChannelSpec("opponents", SPACE_TYPE.CATEGORICAL, low=0, high=1),
#ChannelSpec("food", SPACE_TYPE.CATEGORICAL, low=0, high=1),
#ChannelSpec("obstacles", SPACE_TYPE.CATEGORICAL, low=0, high=1)

obs_space = Box(0, 2048, shape=(12, 12, 4), dtype=np.uint8)
mcbox_spec = MCBoxSpec(CHANNEL_SPECS, obs_space, channel_axis=-1)

LATENT2TARGET_SPECS = [
    ChannelSpec(VALUE_FUNCTION, SPACE_TYPE.NUMERIC, shape=(1,), low=0, high=1, apply_whitening=True),
    ChannelSpec(CONFIDENCE, SPACE_TYPE.NUMERIC, shape=(1,), low=0, high=1, apply_whitening=True),
    ChannelSpec(GOAL_COND, SPACE_TYPE.NUMERIC, shape=(1,), low=-1, high=1, apply_whitening=True),
    ChannelSpec(RISKINESS, SPACE_TYPE.NUMERIC, shape=(1,), low=-5, high=5, apply_whitening=True),
    ChannelSpec(INCONGRUITY, SPACE_TYPE.NUMERIC, shape=(1,), low=-1, high=1, apply_whitening=True)
]


CAML_ROOT = Path(PVAE_ROOT, "..", "..")
DATA_ROOT = Path(CAML_ROOT, "datasets", "canniballs", "interestingness", "v2")
DEFAULT_DATA_FPATH = Path(DATA_ROOT, "interaction_data.pkl.gz")
DEFAULT_CSV_FPATH = Path(DATA_ROOT, "interestingness.csv.gz")

# Adjusted to point to what should be a better model
DEFAULT_MODEL_FPATH = Path(CAML_ROOT, "imago", "models", "canniballs_v2a", "model.pt")

def load_data(data_fpath=DEFAULT_DATA_FPATH, intr_csv_fpath=DEFAULT_CSV_FPATH,
              tt_splitat=0.8):
    """
    :param:tt_splitat:Percentage of rollouts to split train/test at
    """ 
    rollouts = load_object(data_fpath)
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

    for rollout_num, (rid, rollout) in tqdm(enumerate(rollouts.items())):
        """
        Unbox the Rollout and Interaction Data
        """
        rid = int(rid) # CSV lookup uses longs as rollout IDs
        intr_df = vars_df[vars_df.Rollout == rid]  # TODO: Assert these exist
        O = rollout.data.observation  # (Frames, ... )
        R = rollout.data.reward
        V = rollout.data.value
        assert len(intr_df) == len(O) == len(R) == len(V)

        # Unroll the episode into individual instances
        Os.append(O)
        targets[VALUE_FUNCTION].append(V)
        for colname in [CONFIDENCE, GOAL_COND, RISKINESS, INCONGRUITY]:
            arr = intr_df[colname].to_numpy().reshape((-1, 1))  # Padd with a column value so vstack works
            targets[colname].append(arr)
        # Identify episode split point
        if rollout_num < tt_rollout_index:
            tt_eidx += len(O)

    assert tt_eidx > 0 
    Os_train, Os_test = np.vstack(Os[0:tt_rollout_index]), np.vstack(Os[tt_rollout_index:])
    targets_train, targets_test = {}, {}
    for k, v in targets.items():
        targets_train[k] = np.vstack(v[0:tt_rollout_index])
        targets_test[k] = np.vstack(v[tt_rollout_index:])
    train_dataset = DictTensorDataset(Os_train, targets_train)
    test_dataset = DictTensorDataset(Os_test, targets_test)
    return train_dataset, test_dataset


def load_model(model_path=DEFAULT_MODEL_FPATH, device="cpu"):
    model = PVaeModel(mcbox_spec, latent2target_specs=LATENT2TARGET_SPECS,
                      device=device).to(device)
    load_checkpoint(model_path, model, None)
    return model


def render_canniballs(O, cell_width=10):
    """ Convenience routine for rendering Canniballs scenes"""
    return imago.viz.render(O, mcbox_spec, cell_width=cell_width)