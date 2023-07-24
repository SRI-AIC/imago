ROOT_DIR = "."

"""
Similar to train.py, but now selects certain combinations of C that should never be observed in the training set.

These will then be used for the test set.

"""

from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

from imago_prev.pysc2.render import UnitTypeRenderer

from sc2recorder.utils import set_random_seed, ensure_torch, ensure_numpy

from imago_prev.models.supervised.datasets import *
from imago_prev.models.supervised.supervised_condvaes import *
from imago_prev.data.trajectories import *
from imago_prev.models.loader import instance_model
from sc2scenarios.scenarios.simple1 import sample_vec
from absl import logging, app, flags

logging.set_verbosity(logging.INFO)
import pdb


FLAGS = flags.FLAGS
flags.DEFINE_string("device", "cuda:0", "CUDA device")
flags.DEFINE_boolean("debug", False, "Use small data")
flags.DEFINE_boolean("activate_recon_loss", True, "Activate reconstruction loss")
flags.DEFINE_boolean("activate_interp_loss", True, "Activate interpolation loss")
flags.DEFINE_string("kl_penalty_regime", "none", "Type of KL penalty ['full'|'none']")
flags.DEFINE_integer("batch_size", 32, "training batch size")
flags.DEFINE_integer("epochs", 300, "num epochs")
flags.DEFINE_string("model_type", "convcondvae_v2", "Model type (see imago/models/loader.py)")
flags.DEFINE_integer("n_latent", 10, "Size Z")
flags.DEFINE_integer("n_hidden", 10, "Num hidden")
flags.DEFINE_boolean("recon_decode_full_C", False, "Whether to use full C vec for decoding the reconstruction")
flags.DEFINE_boolean("interp_decode_full_C", False, "Whether to use full C vec for decoding the interpolation")
flags.DEFINE_integer("C_heldout_size", 10, "Number of conditions to hold out")

def vec_in(x, M):
    """ Returns if the given vector x is contained in the memory M.
    We expect x to be a row vector (D,), M shape (B, D)"""
    diff = M - x
    #pdb.set_trace()
    return np.any(np.all(diff == 0, axis=1))
    

def main(argv):
    device=FLAGS.device
    set_random_seed()

    C_heldout = []
    num_heldout = FLAGS.C_heldout_size
    if FLAGS.debug:
        num_heldout = 1
    for i in range(num_heldout):
        c_vec = sample_vec()
        tries = 0
        if (i >= 1):
            while (vec_in(c_vec, np.array(C_heldout))):
                print("-----")
                print(i, c_vec, np.array(C_heldout), vec_in(c_vec, np.array(C_heldout)))
                c_vec = sample_vec()
                tries += 1
                if tries % 10 == 0:
                    logging.warning("Warning, failed to sample different C!")
                    break
        C_heldout.append(c_vec)
    if FLAGS.debug:
        # Pull in a known one, since small debug sets may not overlap
        C_heldout.append(np.array([0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0.]))
    C_heldout = np.array(C_heldout)
    logging.info("C heldouts size={}".format(len(C_heldout)))
    for idx, c_vec in enumerate(C_heldout):
        logging.info("Heldout #{}: {}".format(idx, vec2var(c_vec)))
    
    def is_training_fn(Xs, Ys, Cs):
        c = Cs[0,:]
        return not(vec_in(c, C_heldout))
    
    def is_test_fn(Xs, Ys, Cs):
        c = Cs[0,:]
        return vec_in(c, C_heldout)
        
    if FLAGS.debug:
        all_trajs_dir = "{}/data/tensor_replays/small_train".format(ROOT_DIR)
    else:
        all_trajs_dir = "{}/data/tensor_replays/all".format(ROOT_DIR)

    src_traj_ds = TensorTrajectoryDataset(all_trajs_dir)
    train_traj_ds = src_traj_ds.filter(is_training_fn)
    valid_traj_ds = src_traj_ds.filter(is_test_fn)
        
    #train_traj_ds = TensorTrajectoryDataset(all_trajs_dir, filter_fn=None)
    #train_traj_ds = TensorTrajectoryDataset(all_trajs_dir, filter_fn=is_training_fn)
    train_first_n_ds = FirstObsDataset(train_traj_ds)
    train_perturb_ds = PerturbDataset(train_first_n_ds)
    print("Training Trajectory DS size={}, size perturb={}".format(len(train_traj_ds), len(train_perturb_ds)))
    #print(train_traj_ds[0][2])

    #valid_traj_ds = TensorTrajectoryDataset(all_trajs_dir, filter_fn=None)
#    valid_traj_ds = TensorTrajectoryDataset(all_trajs_dir, filter_fn=is_test_fn)
    valid_first_n_ds = FirstObsDataset(valid_traj_ds)
    valid_perturb_ds = PerturbDataset(valid_first_n_ds)
    print("Validation Trajectory DS size={}, size perturb={}".format(len(valid_traj_ds), len(valid_perturb_ds)))
    
    train_dataloader = DataLoader(train_perturb_ds, batch_size=FLAGS.batch_size)
    test_dataloader = DataLoader(valid_perturb_ds, batch_size=FLAGS.batch_size)

    metadata = train_traj_ds.metadata
    N_DISTINCT_UNITS = metadata.N_DISTINCT_UNITS
    OBS_WIDTH, OBS_HEIGHT = metadata.OBS_WIDTH, metadata.OBS_HEIGHT
    N_COND = metadata.N_COND
    N_INPUT = OBS_WIDTH * OBS_HEIGHT * N_DISTINCT_UNITS

    print(N_COND, N_DISTINCT_UNITS, N_INPUT)

    renderer = UnitTypeRenderer(metadata)

    model = instance_model(FLAGS.model_type, metadata, N_LATENT=FLAGS.n_latent, N_HIDDEN=FLAGS.n_hidden,
                           device=FLAGS.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_params = {
        'model_type' : FLAGS.model_type,
        'n_latent': FLAGS.n_latent,
        'n_hidden': FLAGS.n_hidden,
        'activate_recon_loss': FLAGS.activate_recon_loss,
        'activate_interp_loss': FLAGS.activate_interp_loss,
        'kl_penalty_regime': FLAGS.kl_penalty_regime,
        "recon_decode_full_C": FLAGS.recon_decode_full_C,
        "interp_decode_full_C": FLAGS.interp_decode_full_C
    }

    OUT_ROOT = "output/supervised/separated_{}".format(FLAGS.C_heldout_size)
    os.makedirs(OUT_ROOT, exist_ok=True)

    model_name = "-".join(["{}={}".format(k,v) for k,v in train_params.items()])
    logging.info("Model name={}".format(model_name))
    train_model_2head(model, optimizer, train_dataloader, test_dataloader, renderer,
                      epochs=FLAGS.epochs,
                      device=device, save_model_dir="{}/{}".format(OUT_ROOT, model_name), **train_params)

if __name__ == "__main__":
    app.run(main)