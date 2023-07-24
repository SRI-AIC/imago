import torch
import numpy as np
from pysc2.env.environment import TimeStep
from pysc2.env.sc2_env import AgentInterfaceFormat
from pysc2.lib.actions import FunctionCall
from imago_prev.models.sequential.pets.converters.reaver_vae_converter import ReaverVAESampleConverter, _featurize_obs
from imago_prev.models.behav.rb_perturb import RBPerturbModel
from imago_prev.models.behav.reaver_behav import REAVER_ACT_SPECS, ensure_numpy

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

MAX_REWARD = 300
DETERMINISTIC = False  # whether VAE is deterministic, ie always return mean value


class RBVAESampleConverter(ReaverVAESampleConverter):
    """
    A converter from agent observations to latent observations of a VAE model capable of reconstructing both the
    observations and the policy targets (value and prob. distribution over actions).
    """

    def __init__(self, agent_interface_format, vae_model, action_set, action_spatial_dim,
                 deterministic=DETERMINISTIC, max_reward=MAX_REWARD):
        """
        Creates a new reaver converter.
        :param AgentInterfaceFormat agent_interface_format: the agent's pySC2 interface format.
        :param RBPerturbModel vae_model: a trained SC2 VAE model used to convert observations into latent representations.
        :param str action_set: named action set.
        :param int action_spatial_dim: dimension of spatial actions.
        :param bool deterministic: whether the VAE is deterministic, i.e., always returns the mean value.
        :param float max_reward: the maximum reward for this task, for normalization purposes.
        """
        super().__init__(agent_interface_format, vae_model.rb_model.vae_model,
                         action_set, action_spatial_dim, deterministic, max_reward)

        self.rb_vae_model = vae_model
        self.behavior_model = vae_model.rb_model.behav_model

        # additional components provided by behavior in full forward mode
        self.value = np.zeros((1, 1))
        self.action_probs = [np.zeros(np.prod(act_spec_shape)) for _, act_spec_shape in REAVER_ACT_SPECS]

    def from_agent_observations(self, sc2_observations, sc2_actions):
        """
        Converts a batch of pySC2 observation into appropriate observation, action, reward numpy arrays.
        :param list[TimeStep] sc2_observations: the observation in pySC2 features form.
        :param list[list[FunctionCall]] sc2_actions: list of actions executed by the agent between the previous
        observation and the current observation.
        :rtype: (np.ndarray, np.ndarray, np.ndarray)
        :return: a tuple containing the observations, actions, and rewards in an array format. The observations are a
        tuple containing the latent sample, latent mean and latent log variance of the VAE model.
        """

        def normalize(dist):
            dist = ensure_numpy(dist)
            return dist / np.sum(dist)

        # gets observation part
        # create a suitable dict structure to sample the VAE model
        vae_obs = _featurize_obs(sc2_observations)

        # samples the VAE and gets the latent structure (encoding)
        x1 = self._vae_model.sc2_mf_in(vae_obs)
        y1, z_mu, z_logvar, z, a = self._vae_model.vae_model(x1, deterministic=self._deterministic)
        self.z_mu = ensure_numpy(z_mu)
        self.z_logvar = ensure_numpy(z_logvar)

        # converts action
        sc2_actions = np.array([self._reverse_action(sc2_actions[i][0])
                            if len(sc2_actions[i]) > 0 else [0] * len(self._act_wrapper.spec.spaces)
                                for i in range(len(sc2_actions))])

        rewards = np.array([sc2_observations[i].reward / self._max_reward for i in range(len(sc2_observations))]).reshape((-1, 1))

        # gets policy part
        b_y, v = self.behavior_model.forward_Z(z)
        self.value = ensure_numpy(v).flatten()
        self.action_probs = [np.array([normalize(b_y[t][a]) for t in range(len(b_y))])
                             for a in range(len(REAVER_ACT_SPECS))]

        z = ensure_numpy(z)

        return z, sc2_actions, rewards

    def to_agent_policy(self, observations):
        """
        Converts a batch of numpy latent observations to corresponding policy value and action dist. using behavior model.
        :param np.ndarray observations: the numpy array with the observation in the VAE's latent representation.
        :rtype: (np.ndarray, list[dict[str, np.ndarray]])
        :return: a tuple containing the state value and action probability distributions reconstructed from each
        latent observation.
        """

        z = torch.as_tensor(observations, dtype=torch.float32)
        b_y, value = self.behavior_model.forward_Z(z)  # Get behavior predictions
        action_dists = self.rb_vae_model._format_as_dists(b_y)

        return value, action_dists
