import os
import logging
import numpy as np
import tensorflow as tf
import gtimer as gt
from tensorflow.python.client import device_lib
from mbpo.models.constructor import format_samples_for_training
from mbpo.models import bnn
from mbpo.models.fc import FC
from imago_prev.models.sequential.pets.plot import plot_evolution

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

OBS_STR = 'observations'
ACT_STR = 'actions'
NEXT_OBS_STR = 'next_observations'
RWD_STR = 'rewards'

ACTIVATION = 'swish'  # the activation function applied on the outputs of each layer
FIRST_WEIGHT_DECAY = 0.000025  # 0.0001  # the rate of weight decay applied to the weights of the first layer
MIDDLE_WEIGHT_DECAYS = [0.00005, 0.000075, 0.000075]  # the weight decays of the middle layers
LAST_WEIGHT_DECAY = 0.0001  # 0.0005  # the rate of weight decay applied to the weights of the last layer


def setup_tf(seed=0):
    """
    Sets up a tensorflow session for Predictive Ensemble training/testing.
    :param int seed: the seed to initialize tensorflow.
    :rtype: (tf.Session, bool)
    :return: a tuple containing the created session and whether the current environment has a GPU device.
    """
    tf.set_random_seed(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
    tf.keras.backend.set_session(session)
    session = tf.keras.backend.get_session()
    session.__enter__()
    devices = device_lib.list_local_devices()
    logging.info('Available computing devices: {}'.format(devices))
    return session, any([dev.device_type == 'GPU' for dev in devices])


def get_samples_dict(observations, actions, rewards, next_observations):
    """
    Packages the given lists of samples into a dictionary to be used by the MBPO library.
    :param list[np.ndarray] or np.ndarray observations: the list of observations.
    :param list[np.ndarray] or np.ndarray actions: the list of actions.
    :param list[np.ndarray] or np.ndarray rewards: the list of rewards.
    :param list[np.ndarray] or np.ndarray next_observations: the list of next observations.
    :rtype: dict[str, np.ndarray]
    :return: a dictionary containing a numpy array containing the samples for each type of information.
    """
    return {OBS_STR: np.asarray(observations),
            ACT_STR: np.asarray(actions),
            RWD_STR: np.asarray(rewards),
            NEXT_OBS_STR: np.asarray(next_observations)}


class BNN(bnn.BNN):
    """
    A wrapper for the predictive ensemble (PE) of probabilistic models.
    """

    def __init__(self, model_dir, obs_dim, act_dim, rew_dim=1, load=False,
                 hidden_dim=200, num_networks=7, num_elites=5, learning_rate=0.001,
                 max_epochs=None, batch_size=32, holdout_ratio=0.2, max_model_t=None,
                 session=None):
        """
        Creates a new PE or loads on from file.
        :param str model_dir: path to directory from which model will be loaded or saved by default.
        :param int obs_dim: the dimensionality of the input observations.
        :param int act_dim: the dimensionality of the input actions.
        :param int rew_dim:  the dimensionality of the output rewards.
        :param bool load: whether to load the model from the model directory.
        :param int hidden_dim: the number of hidden units in each layer of the model.
        :param int num_networks: the number of networks in the prediction ensemble model.
        :param int num_elites: the number of networks used for elite (best) selection.
        :param float learning_rate: the optimizer's learning rate.
        :param int max_epochs: max. number of epochs (full network passes) that will be done in each training step. `None` for no limit.
        :param int batch_size: the mini-batch size to be used for training.
        :param float holdout_ratio: portion of samples used for testing, ie computing the loss at each train step.
        :param int max_model_t: the maximum number of seconds allowed for each model training step. `None` for no limit.
        :param tf.Session session: the TF session in which to create PE model, or `None` to create a new one.
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rwd_dim = rew_dim

        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._holdout_ratio = holdout_ratio
        self._max_model_t = max_model_t

        # stats
        self.model_metrics = []
        self.total_samples = 0

        # initializes BNN
        if load:
            super().__init__(dict(name='BNN_0', model_dir=model_dir, load_model=True,
                                  num_networks=-1, num_elites=-1, sess=session))
        else:
            super().__init__(dict(name='BNN', model_dir=model_dir, load_model=False,
                                  num_networks=num_networks, num_elites=num_elites, sess=session))

            # add layers
            self.add(FC(hidden_dim, input_dim=obs_dim + act_dim,
                        activation=ACTIVATION, weight_decay=FIRST_WEIGHT_DECAY))
            for weight_decay in MIDDLE_WEIGHT_DECAYS:
                self.add(FC(hidden_dim, activation=ACTIVATION, weight_decay=weight_decay))
            self.add(FC(obs_dim + rew_dim, weight_decay=LAST_WEIGHT_DECAY))

        # finalize model
        self.finalize(tf.train.AdamOptimizer, {'learning_rate': learning_rate})

    @property
    def train_steps(self):
        """
        Gets the total number of training steps that this model has been through so far.
        :rtype: int
        :return: the number of training steps.
        """
        return len(self.model_metrics)

    @classmethod
    def load(cls, model_dir, obs_dim, act_dim, rew_dim=1, session=None):
        """
        Loads a predictive ensemble model from the given directory
        :param str model_dir: path to directory from which the model will be loaded.
        :param int obs_dim: the dimensionality of the input observations.
        :param int act_dim: the dimensionality of the input actions.
        :param int rew_dim:  the dimensionality of the output rewards.
        :param tf.Session session: the TF session in which to create PE model, or `None` to create a new one.
        :rtype: BNN
        :return: the loaded predictive model.
        """
        model = BNN(model_dir, obs_dim, act_dim, rew_dim, True, session=session)

        # check dimensions
        input_dim = model.layers[0].get_input_dim()
        assert input_dim == obs_dim + act_dim, \
            'Input dimensions provided ({}+{}) differs from loaded model: {}'.format(obs_dim, act_dim, input_dim)
        output_dim = model.layers[-1].get_output_dim() / 2
        assert output_dim == obs_dim + rew_dim, \
            'Output dimensions provided ({}+{}) differs from loaded model: {}'.format(rew_dim, obs_dim, output_dim)

        return model

    def train_model(self, observations, actions, rewards, next_observations):
        """
        Performs one training step of the PE model given an experience batch.
        :param list[np.ndarray] or np.ndarray observations: the list of observations.
        :param list[np.ndarray] or np.ndarray actions: the list of actions.
        :param list[np.ndarray] or np.ndarray rewards: the list of rewards.
        :param list[np.ndarray] or np.ndarray next_observations: the list of next observations.
        :return:
        """

        # adapted from mbpo/algorithms/mbpo.py
        samples = get_samples_dict(observations, actions, rewards, next_observations)
        train_inputs, train_outputs = format_samples_for_training(samples)
        metrics = self.train(
            train_inputs, train_outputs,
            batch_size=self._batch_size, max_epochs=self._max_epochs,
            holdout_ratio=self._holdout_ratio, max_t=self._max_model_t, timer=gt)

        self.total_samples += train_inputs.shape[0]

        times = gt.get_times()
        metrics.update(times.stamps.cum)
        metrics['total_time'] = times.total
        gt.reset()
        self.model_metrics.append(metrics)

    def get_rollouts(self, observations, actions, actor_func, rollout_length=1, deterministic=True):
        """
        Gets a predicted rollout from the ensemble model.
        :param list[np.ndarray] or np.ndarray observations: the list of observations from which to rollout.
        :param list[np.ndarray] or np.ndarray actions: the list of actions from which to rollout.
        :param actor_func: a function (`np.ndarray`: `np.ndarray`) that given a set of observations returns the
        corresponding actions to be executed.
        :param int rollout_length: the length of the rollout
        :param bool deterministic: whether to make deterministic (`True`) or stochastic (`False`) samples.
        :rtype: (np.ndarray, np.ndarray)
        :return: a tuple containing:
            - the predicted rewards, an array of shape (length, sample+mean+var (3), num_nets, batch_size, rwd_dim)
            - the predicted observations, an array of shape (length, sample+mean+var (3), num_nets, batch_size, obs_dim)
        """

        # check dims
        assert len(observations.shape) == 2 and observations.shape[1] == self.obs_dim, \
            'Observation dims provided {} do not match model\'s dims: {}'.format(observations.shape[-1], self.obs_dim)
        assert len(actions.shape) == 2 and actions.shape[1] == self.act_dim, \
            'Action dims provided {} do not match model\'s dims: {}'.format(actions.shape[-1], self.act_dim)

        # initial obs and actions are the same for all predictors in the ensemble
        batch_size = observations.shape[0]
        observations = np.tile(observations, (self.num_nets, 1, 1))
        actions = np.tile(actions, (self.num_nets, 1, 1))

        # rollout tuple shape: [length, sample + mean + var (3), num_nets, batch_size, rwd_dim + obs_dim]
        rollout = np.zeros((rollout_length, 3, self.num_nets, batch_size, self.rwd_dim + self.obs_dim))

        dummy_rwds = np.zeros((self.num_nets, batch_size, self.rwd_dim))
        dummy_next_obs = np.zeros_like(observations)
        for i in range(rollout_length):

            # prepares samples
            # - input: [num_nets, batch_size, obs_dim + act_dim]
            samples = get_samples_dict(observations, actions, dummy_rwds, dummy_next_obs)
            inputs, _ = format_samples_for_training(samples)

            # sample predictive model, with shapes:
            #   - output: [num_nets, batch_size, rwd_dim + obs_dim]
            model_means, model_vars = self.predict(inputs, factored=True)
            model_means[:, :, self.rwd_dim:] += observations  # add prev obs as model computes obs difference

            # if deterministic, take the mean, otherwise sample from Gaussian
            if deterministic:
                model_samples = model_means
            else:
                model_samples = model_means + np.random.normal(size=model_means.shape) * np.sqrt(model_vars)

            # stores data
            rollout[i, 0, :, :, :] = model_samples
            rollout[i, 1, :, :, :] = model_means
            rollout[i, 2, :, :, :] = model_vars

            # prepares next step and gets actions from actor
            if i < rollout_length - 1:
                observations = model_samples[:, :, self.rwd_dim:]
                actions = actor_func(observations)

        # splits rollout data and returns
        return rollout[:, :, :, :, :self.rwd_dim], rollout[:, :, :, :, self.rwd_dim:]

    def predict_from_observations(self, observations, actions, deterministic=True):
        """
        Gets a next-step prediction from the ensemble model.
        :param list[np.ndarray] or np.ndarray observations: the list of observations from which to rollout.
        :param list[np.ndarray] or np.ndarray actions: the list of actions from which to rollout.
        :param bool deterministic: whether to make deterministic (`True`) or stochastic (`False`) samples.
        :rtype: (np.ndarray, np.ndarray)
        :return: a tuple containing:
            - the predicted rewards, an array of shape (sample+mean+var (3), num_nets, batch_size, rwd_dim)
            - the predicted observations, an array of shape (sample+mean+var (3), num_nets, batch_size, obs_dim)
        """
        next_rwds, next_obs = self.get_rollouts(observations, actions, None, 1, deterministic)
        return next_rwds[0], next_obs[0]

    def save_model(self):
        """
        Saves the model to the output directory.
        :rtype: str
        :return: the path to the files (prefix) where the model was saved.
        """
        self.save(None, 0)
        return os.path.join(self.model_dir, 'BNN_0')

    def save_stats(self):
        """
        Saves statistics regarding the PE model training to the output directory.
        :return:
        """
        for metric in self.model_metrics[0]:
            data = np.array([[self.model_metrics[i][metric] for i in range(len(self.model_metrics))]])
            logging.info('[PE Trainer] Mean {}: {}'.format(metric.lower(), np.mean(data)))
            img_file = os.path.join(self.model_dir, 'evo-{}.pdf'.format(metric.replace('_', '-').lower()))
            plot_evolution(data, [''], metric.replace('_', ' ').title(), x_label='Training Steps',
                           output_img=img_file, num_points=100)

    def _set_state(self):
        # avoid this step
        return
