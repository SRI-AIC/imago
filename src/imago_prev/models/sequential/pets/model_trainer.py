import logging
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from imago_prev.models.sequential.pets.bnn import BNN
from imago_prev.models.sequential.pets.converters import SampleConverter

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class ProbabilisticEnsembleTrainer(mp.Process):
    """
    Represents a world dynamics model trainer using the Probabilistic Ensemble (PE) approach.
    """

    def __init__(self, converter, output_dir, max_samples_buffer,
                 rew_dim=1, hidden_dim=200, num_networks=7, num_elites=5, learning_rate=0.001,
                 max_epochs=None, batch_size=32, holdout_ratio=0.2, max_model_t=None, session=None):
        """
        Creates a new Predictive Ensemble model trainer.
        :param SampleConverter converter: the converter of samples to numpy representations.
        :param str output_dir: path to directory from which model will be loaded, and saved by default.
        :param int max_samples_buffer: maximum size of the rotating training buffer.
        :param int rew_dim: the size of the reward arrays.
        :param int hidden_dim: the number of hidden units in each layer of the model.
        :param int num_networks: the number of networks in the prediction ensemble model.
        :param int num_elites: the number of networks used for elite (best) selection.
        :param float learning_rate: the optimizer's learning rate.
        :param int max_epochs: max. number of epochs (full network passes) that will be done in each training step. `None` for no limit.
        :param int batch_size: the minibatch size to be used for training.
        :param float holdout_ratio: portion of samples used for testing, ie computing the loss at each train step.
        :param int max_model_t: the maximum number of seconds allowed for each model training step. `None` for no limit.
        :param tf.Session session: the TF session in which to create PE model, or `None` to create a new one.
        """
        super().__init__()
        self._converter = converter
        self._output_dir = output_dir
        self._max_samples_buffer = max_samples_buffer

        # samples buffer
        self._observations = None
        self._actions = None
        self._rewards = None
        self._next_observations = None

        obs_dim = converter.observation_dim
        act_dim = converter.action_dim

        logging.info('[PE Trainer] Creating new BNN model with observation dim: {}, action dim: {}, '
                     'reward dim: {}, hidden dim: {}'.format(obs_dim, act_dim, rew_dim, hidden_dim))

        # create a new model with the several layers
        self._pe_model = BNN(output_dir, obs_dim, act_dim, rew_dim, False,
                             hidden_dim, num_networks, num_elites, learning_rate,
                             max_epochs, batch_size, holdout_ratio, max_model_t, session)

    def train(self, samples):
        """
        Trains the dynamics model using the given samples.
        """

        # converts batch of agent observations
        _, agent_obs, agent_actions = samples
        obs, actions, rewards = self._converter.from_agent_observations(agent_obs, agent_actions)

        # adds converted elements to buffers
        self._observations = obs[:-1] if self._observations is None else \
            np.concatenate((self._observations, obs[:-1]))
        self._actions = actions[1:] if self._actions is None else np.concatenate((self._actions, actions[1:]))
        self._rewards = rewards[1:] if self._rewards is None else np.concatenate((self._rewards, rewards[1:]))
        self._next_observations = obs[1:] if self._next_observations is None else \
            np.concatenate((self._next_observations, obs[1:]))

        # check max buffer size
        if len(self._observations) > self._max_samples_buffer:
            n_remove_samples = len(self._observations) - self._max_samples_buffer
            self._observations = self._observations[n_remove_samples:]
            self._actions = self._actions[n_remove_samples:]
            self._rewards = self._rewards[n_remove_samples:]
            self._next_observations = self._next_observations[n_remove_samples:]

        # train model
        logging.info('[PE Trainer] {}. Training model with {} newly-collected samples (total {})...'.format(
            self._pe_model.train_steps, len(agent_obs) - 1, len(self._observations)))

        self._pe_model.train_model(self._observations, self._actions, self._rewards, self._next_observations)

        logging.info('[PE Trainer] {}. Loss: {}'.format(
            self._pe_model.train_steps - 1, self._pe_model.model_metrics[-1]['val_loss']))

    def end_train(self):
        """
        Finished training the model. Saves the model and prints some statistics.
        """
        logging.info('[PE Trainer] Finished training for {} steps, {} total samples...'.format(
            self._pe_model.train_steps, self._pe_model.total_samples))

        self.save_model()

        logging.info('[PE Trainer] Saving statistics...')
        self._pe_model.save_stats()

    def save_model(self):
        """
        Saves the model to the predefined directory.
        """
        file_path = self._pe_model.save_model()
        logging.info('[PE Trainer] Saved PE model to {}'.format(file_path))
        self._pe_model.save_stats()
