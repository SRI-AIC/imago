ROOT_DIR = "."

from torch import nn, optim
from torch.utils.data import DataLoader


from imago_prev.pysc2.render import UnitTypeRenderer

from sc2recorder.utils import set_random_seed, ensure_torch, ensure_numpy

from imago_prev.models.supervised.datasets import *
from imago_prev.models.supervised.supervised_condvaes import *
from imago_prev.data.trajectories import *
from imago_prev.models.loader import instance_model
from absl import logging, app, flags

logging.set_verbosity(logging.INFO)


FLAGS = flags.FLAGS
flags.DEFINE_string("device", "cuda:0", "CUDA device")
flags.DEFINE_boolean("debug", False, "Use small data")
flags.DEFINE_boolean("activate_recon_loss", True, "Activate reconstruction loss")
flags.DEFINE_boolean("activate_interp_loss", True, "Activate interpolation loss")
flags.DEFINE_string("kl_penalty_regime", "full", "Type of KL penalty ['full'|'none']")
flags.DEFINE_integer("batch_size", 32, "training batch size")
flags.DEFINE_integer("epochs", 300, "num epochs")
flags.DEFINE_string("model_type", "convcondvae_v2", "Model type (see imago/models/loader.py)")
flags.DEFINE_integer("n_latent", 10, "Size Z")
flags.DEFINE_integer("n_hidden", 10, "Num hidden")
flags.DEFINE_boolean("recon_decode_full_C", False, "Whether to use full C vec for decoding the reconstruction")
flags.DEFINE_boolean("interp_decode_full_C", False, "Whether to use full C vec for decoding the interpolation")
def main(argv):
    device=FLAGS.device
    set_random_seed()

    if FLAGS.debug:
        train_dir = "{}/data/tensor_replays/small_train".format(ROOT_DIR)
    else:
        train_dir = "{}/data/tensor_replays/train".format(ROOT_DIR)
    valid_dir = "{}/data/tensor_replays/valid".format(ROOT_DIR)

    train_traj_ds = TensorTrajectoryDataset(train_dir)
    train_first_n_ds = FirstObsDataset(train_traj_ds)
    train_perturb_ds = PerturbDataset(train_first_n_ds)

    valid_traj_ds = TensorTrajectoryDataset(valid_dir)
    valid_first_n_ds = FirstObsDataset(valid_traj_ds)
    valid_perturb_ds = PerturbDataset(valid_first_n_ds)

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

    os.makedirs("output/supervised", exist_ok=True)

    model_name = "-".join(["{}={}".format(k,v) for k,v in train_params.items()])
    logging.info("Model name={}".format(model_name))
    train_model_2head(model, optimizer, train_dataloader, test_dataloader, renderer,
                      epochs=FLAGS.epochs,
                      device=device, save_model_dir="output/supervised/{}".format(model_name), **train_params)

if __name__ == "__main__":
    app.run(main)