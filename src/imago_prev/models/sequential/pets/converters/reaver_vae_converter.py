import copy
import reaver
import torch
import numpy as np
import imago_prev.models.semframe.plot as splot
from PIL import Image
from pysc2.env import mock_sc2_env
from pysc2.env.environment import TimeStep
from pysc2.env.sc2_env import AgentInterfaceFormat
from pysc2.lib.actions import FunctionCall, numpy_to_python, FUNCTIONS
from reaver.envs.sc2 import ActionWrapper, ObservationWrapper, ReaverStateActionSpace
from imago_prev.models.semframe.model_remapper import SC2VAE_remapper
from imago_prev.models.sequential.pets.converters import SampleConverter

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

ACTION_SET = {
    'minimal': reaver.envs.sc2.MINIMAL_ACTION_SET,
    'screen': reaver.envs.sc2.MINIMAL_SCREEN_ACTION_SET}

OBS_FEATURES = {'minimal': reaver.envs.sc2.MINIMAL_FEATURES,
                'vae': reaver.envs.sc2.VAE_FEATURES,
                'vae2': reaver.envs.sc2.VAE_FEATURES_2,
                'screen': reaver.envs.sc2.MINIMAL_SCREEN_FEATURES}

SC2_FEATURE_NAMES = ['feature_screen', 'feature_minimap', 'available_actions', 'player']

MAX_REWARD = 300
DETERMINISTIC = False  # whether VAE is deterministic, ie always return mean value


class ReaverVAESampleConverter(SampleConverter):
    """
    A converter from agent observations to latent observations of a VAE model suitable for reaver RL models.
    """

    def __init__(self, agent_interface_format, vae_model, action_set, action_spatial_dim,
                 deterministic=DETERMINISTIC, max_reward=MAX_REWARD):
        """
        Creates a new reaver converter.
        :param AgentInterfaceFormat agent_interface_format: the agent's pysc2 interface format.
        :param SC2VAE_remapper vae_model: a trained SC2 VAE model used to convert observations into latent representations.
        :param str action_set: named action set.
        :param int action_spatial_dim: dimension of spatial actions.
        :param bool deterministic: whether the VAE is deterministic, i.e., always returns the mean value.
        :param float max_reward: the maximum reward for this task, for normalization purposes.
        """
        super().__init__(agent_interface_format)
        self._vae_model = vae_model
        self._sa_space = _create_sa_space('vae2', action_set, agent_interface_format, action_spatial_dim)
        self._act_wrapper = self._sa_space.act_wrapper
        self._obs_dim = vae_model.vae_model.N_LATENT
        self._act_dim = _get_dims(self._act_wrapper)
        self._deterministic = deterministic
        self._max_reward = max_reward

        self.z_mu = np.zeros(vae_model.vae_model.N_LATENT)
        self.z_logvar = np.zeros(vae_model.vae_model.N_LATENT)

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

        # create a suitable dict structure to sample the VAE model
        vae_obs = _featurize_obs(sc2_observations)

        # samples the VAE and gets the latent structure (encoding)
        x1 = self._vae_model.sc2_mf_in(vae_obs)
        y1, z_mu, z_logvar, z, a = self._vae_model.vae_model(x1, deterministic=self._deterministic)
        z = z.detach().numpy()
        self.z_mu = z_mu.detach().numpy()
        self.z_logvar = z_logvar.detach().numpy()

        # converts action
        sc2_actions = np.array([self._reverse_action(sc2_actions[i][0])
                            if len(sc2_actions[i]) > 0 else [0] * len(self._act_wrapper.spec.spaces)
                                for i in range(len(sc2_actions))])

        rewards = np.array([sc2_observations[i].reward / self._max_reward
                            for i in range(len(sc2_observations))]).reshape((-1, 1))

        return z, sc2_actions, rewards

    def to_agent_observations(self, observations, rewards, sc2_observations):
        """
        Converts a batch of numpy observations to appropriate pySC2 agent observations.
        :param np.ndarray observations: the numpy array with the observation in the VAE's latent representation.
        :param np.ndarray rewards: the numpy array corresponding to the rewards.
        :param list[TimeStep] sc2_observations: the original or approximate observations in pySC2 features form.
        :rtype: list[TimeStep]
        :return: a list containing the converted agent observations.
        """
        # decodes given latent obs to a dictionary structure
        vae_obs = self._vae_model.fwd_z2dict(torch.as_tensor(observations, dtype=torch.float32))

        # replace feature layers from reconstructed observation
        res_agent_obs = []
        for i, ag_obs in enumerate(sc2_observations):
            observation = copy.deepcopy(ag_obs.observation)  # create hard copy of observation
            for f in SC2_FEATURE_NAMES:
                if f in vae_obs[i]['data']:
                    # if feature layer is in the vae observation, replace where we have the information
                    for k in observation[f]._index_names[0]:
                        if k in vae_obs[i]['data'][f]:
                            observation[f][k] = vae_obs[i]['data'][f][k]

            # create new agent obs (TimeStep) with data replaced (step type and discount are copied)
            res_agent_obs.append(
                TimeStep(ag_obs.step_type, rewards[i].item() * self._max_reward, ag_obs.discount, observation))

        return res_agent_obs

    def to_images(self, sc2_observations, features=None):
        """
        Converts the `feature_screen` layers of a batch of pySC2 observations into image representations.
        :param list[TimeStep] sc2_observations: the observations in pySC2 features form.
        :param list[str] features: the list of screen features to be rendered into the image.
        If set to `None`, all available layers are used.
        :rtype: list[Image.Image]
        :return: a list containing the image renderings for each of the given observations.
        """
        # todo parse components and select only the ones in `features`
        return [splot.render_frames(ag_obs, self._vae_model.components) for ag_obs in _featurize_obs(sc2_observations)]

    @property
    def observation_dim(self):
        return self._obs_dim

    @property
    def action_dim(self):
        return self._act_dim

    def _reverse_action(self, agent_action):

        # reverse of ActionWrapper.__call__() at reaver/envs/sc2/env.py
        action = [0] * len(self._act_wrapper.spec.spaces)

        # get action idx
        fn_id = np.where([f.id == agent_action.function for f in FUNCTIONS])[0]
        fn_id = fn_id[0] if len(fn_id) > 0 else 0
        fn_id_idx = self._act_wrapper.func_ids.index(fn_id)
        action[0] = fn_id_idx

        # get args
        for i, arg_type in enumerate(FUNCTIONS[fn_id].args):
            arg_name = arg_type.name
            if arg_name in self._act_wrapper.args:
                arg = agent_action.arguments[i]
                arg = numpy_to_python(arg)

                # adapted from FunctionCall.init_with_validation at pysc2/lib/actions.py
                if arg_type.values:  # enum
                    arg = list(arg_type.values).index(arg[0])
                elif len(arg) == 1 and isinstance(arg[0], int):  # Allow bare ints.
                    arg = arg[0]
                elif len(arg) > 1:
                    # pysc2 expects spatial coords, but we have flattened => attempt to fix
                    arg = [int(a / self._act_wrapper.spatial_scale - 0.5) for a in arg]
                    arg = arg[1] * self._act_wrapper.action_spatial_dim + arg[0]

                arg_idx = self._act_wrapper.args.index(arg_name) + 1
                action[arg_idx] = arg

        return action


def _featurize_obs(agent_obs):
    # adapted from 'sc2recorder.replay_recorder.RecorderListener._record'
    scene_records = []
    for i in range(len(agent_obs)):
        scene_record = {'meta': {'discount': agent_obs[i].discount,
                                 'reward': agent_obs[i].reward,
                                 'step_type': agent_obs[i].step_type},
                        'data': {}}
        for f in SC2_FEATURE_NAMES:
            if f in agent_obs[i].observation:
                scene_record['data'][f] = agent_obs[i].observation[f]
        scene_records.append(scene_record)

    return scene_records


def _get_dims(wrapper):
    total_size = 0
    for space in wrapper.spec.spaces:
        total_size += 1 if space.categorical else np.prod(space.shape)
    return total_size


def _create_sa_space(obs_features, action_set, agent_interface_format, action_spatial_dim):
    # (adapted from 'reaver/envs/sc2/env.py')
    obs_features = OBS_FEATURES[obs_features]
    action_ids = ACTION_SET[action_set]
    obs_features = obs_features or reaver.envs.sc2.MINIMAL_FEATURES.copy()
    action_ids = action_ids or reaver.envs.sc2.MINIMAL_ACTION_SET[:]

    obs_spatial_dim = agent_interface_format.feature_dimensions.screen[0]
    action_spatial_dim = action_spatial_dim or obs_spatial_dim
    act_wrapper = ActionWrapper(action_spatial_dim, action_ids, obs_spatial_dim=obs_spatial_dim)
    obs_wrapper = ObservationWrapper(obs_features, action_ids)

    mock_env = mock_sc2_env.SC2TestEnv(agent_interface_format=[agent_interface_format])

    act_wrapper.make_spec(mock_env.action_spec())
    obs_wrapper.make_spec(mock_env.observation_spec())
    mock_env.close()

    return ReaverStateActionSpace(
        agent_interface_format, obs_spatial_dim, action_spatial_dim, obs_wrapper, act_wrapper)
