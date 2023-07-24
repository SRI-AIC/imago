import os
import json
import logging
import multiprocessing as mp
from absl import app, flags
from pysc2.lib import features
from sc2recorder.replayer import ReplayProcessRunner
from sc2recorder.utils import change_log_handler
from feature_extractor.util import create_clear_dir
from imago_prev.models.semframe import model_remapper
from imago_prev.models.sequential.pets.converters.reaver_vae_converter import ReaverVAESampleConverter
from imago_prev.models.sequential.pets.model_trainer import ProbabilisticEnsembleTrainer
from imago_prev.models.sequential.pets.bnn import setup_tf
from imago_prev.models.sequential.pets.replay_sampler import ReplaySamplerProcessor

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__desc__ = 'Run SC2 to replay a game and build a world dynamics model. '

MAX_SAMPLES = 2000
TRAIN_FREQ = 200
NUM_NETWORKS = 7  # 3  # 1  # 7
NUM_ELITES = 1  # 5
HIDDEN_DIM = 200  # 500
LEARNING_RATE = 0.0005  # 0.001  # optimizer's learning rate
MAX_EPOCHS = None  # number of epochs (full network passes that will be done in each training step
BATCH_SIZE = 2048  # 1024  # the minibatch size to be used for training
HOLDOUT_RATIO = 0.2  # portion of samples used for testing, ie computing the loss at each train step
SAVE_FREQ = 100
SEED = 0

flags.DEFINE_string('output', 'output', 'Path to the directory in which to save the dynamics model')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results')

flags.DEFINE_string('vae_model', None, 'Path to the root directory of the Conv. VAE model.')

flags.DEFINE_integer('feature_screen_size', 64, 'Resolution for screen feature layers')
flags.DEFINE_integer('feature_minimap_size', 64, 'Resolution for minimap feature layers')
flags.DEFINE_integer('feature_camera_width', 24, 'Width of the feature layer camera')
flags.DEFINE_string('action_set', 'screen', 'reaver named action set')
flags.DEFINE_integer('action_spatial_dim', 16, 'Dimension of spatial actions')

flags.DEFINE_integer('max_samples_buffer', MAX_SAMPLES, 'Maximum size of the rotating samples buffer')
flags.DEFINE_integer('model_train_freq', TRAIN_FREQ, 'Frequency with which to train the model')
flags.DEFINE_integer('model_save_freq', SAVE_FREQ, 'Frequency with which to train the model')
flags.DEFINE_integer('hidden_dim', HIDDEN_DIM, 'The number of hidden units in each layer of the model')
flags.DEFINE_integer('num_networks', NUM_NETWORKS, 'Number of networks in the prediction ensemble model')
flags.DEFINE_integer('num_elites', NUM_ELITES, 'Number of networks used for elite selection')
flags.DEFINE_float('learning_rate', LEARNING_RATE, 'The optimizer\'s learning rate')
flags.DEFINE_integer('max_epochs', MAX_EPOCHS,
                     'Max. number of epochs (full network passes) that will be done in each training step')
flags.DEFINE_integer('batch_size', BATCH_SIZE, 'The minibatch size to be used for training')
flags.DEFINE_float('holdout_ratio', HOLDOUT_RATIO,
                   'Portion of samples used for testing, ie computing the loss at each train step')
flags.DEFINE_integer('max_model_t', None, 'Maximum number of seconds allowed for each model training step')
flags.DEFINE_integer('seed', SEED, 'The seed to initialize the predictive ensemble training.')

flags.mark_flags_as_required(['replays', 'vae_model'])


class _TrainerProcess(mp.Process):

    def __init__(self, samples_queue, vae_model_dir, aif, output_dir, action_set, action_spatial_dim,
                 save_freq=SAVE_FREQ, seed=SEED, **kwargs):
        super().__init__()
        self._seed = seed
        self._samples_queue = samples_queue
        self._vae_model_dir = vae_model_dir
        self._aif = aif
        self._output_dir = output_dir
        self._action_set = action_set
        self._action_spatial_dim = action_spatial_dim
        self._save_freq = save_freq
        self._kwargs = kwargs

    def run(self):
        # setup TF
        session, has_gpu = setup_tf(self._seed)

        # loads VAE model
        vae_model = model_remapper.load_model(self._vae_model_dir, device=None if has_gpu else 'cpu')
        converter = ReaverVAESampleConverter(self._aif, vae_model, self._action_set, self._action_spatial_dim)

        # creates the predictive model and starts processing samples to train it
        trainer = ProbabilisticEnsembleTrainer(converter, self._output_dir, session=session, **self._kwargs)
        self._train_loop(trainer)

    def _train_loop(self, trainer):

        # starts processing data from all replays
        i = 1
        while True:

            # waits for samples to arrive
            logging.info('[Trainer Process] Waiting for samples...')
            samples = self._samples_queue.get()
            if samples is None:
                break

            # train
            trainer.train(samples)
            self._samples_queue.task_done()  # signals done to someone who is waiting

            # checks save
            if i % self._save_freq == 0:
                trainer.save_model()
            i += 1

        logging.info('[Trainer Process] Finished after {} iterations!'.format(i))
        trainer.end_train()


def main(unused_argv):
    args = flags.FLAGS

    # checks output dir and files
    create_clear_dir(args.output, args.clear)
    change_log_handler(os.path.join(args.output, 'trainer.log'), args.verbosity)

    # save args
    with open(os.path.join(args.output, 'args.json'), 'w') as fp:
        json.dump({k: args[k].value for k in args}, fp, indent=4)

    # creates agent interface format and sample converter
    aif = features.parse_agent_interface_format(
        camera_width_world_units=args.feature_camera_width,
        use_camera_position=True,
        use_feature_units=True,
        feature_screen=args.feature_screen_size,
        feature_minimap=args.feature_minimap_size)

    # create and runs the model trainer
    replay_samples_queue = mp.JoinableQueue(int(args.max_samples_buffer / args.model_train_freq))
    trainer_process = _TrainerProcess(
        replay_samples_queue, args.vae_model, aif, args.output, args.action_set, args.action_spatial_dim,
        args.model_save_freq, args.seed,
        max_samples_buffer=args.max_samples_buffer,
        rew_dim=1,
        hidden_dim=args.hidden_dim,
        num_networks=args.num_networks,
        num_elites=args.num_elites,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        holdout_ratio=args.holdout_ratio,
        max_model_t=args.max_model_t)
    trainer_process.start()

    # creates and runs the replay processor
    sample_processor = ReplaySamplerProcessor(aif, replay_samples_queue, args.model_train_freq, args.step_mul)
    replayer_processor = ReplayProcessRunner(args.replays, sample_processor, args.replay_sc2_version, args.parallel)
    replayer_processor.run()

    # terminates sample processor
    replay_samples_queue.put(None)
    trainer_process.join()
    logging.info('Finished')


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == '__main__':
    app.run(main)
