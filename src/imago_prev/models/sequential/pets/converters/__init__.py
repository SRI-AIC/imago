import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
from pysc2.env.environment import TimeStep
from pysc2.lib.actions import FunctionCall

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class SampleConverter(ABC):
    """
    An abstract class for converters of pySC2 observations into other representations useful for model training.
    """

    def __init__(self, agent_interface_format):
        """
        Creates a new converter.
        :param AgentInterfaceFormat agent_interface_format: the agent's pySC2 interface format.
        """
        self.agent_interface_format = agent_interface_format

    @abstractmethod
    def from_agent_observations(self, sc2_observations, sc2_actions):
        """
        Converts a batch of pySC2 observation into appropriate observation, action, reward numpy arrays.
        :param list[TimeStep] sc2_observations: the observation in pySC2 features form.
        :param list[list[FunctionCall]] sc2_actions: list of actions executed by the agent between the previous
        observation and the current observation.
        :rtype: (np.ndarray,np.ndarray,np.ndarray)
        :return: a tuple containing the observations, actions, and rewards in an array format.
        """
        pass

    @abstractmethod
    def to_agent_observations(self, observations, rewards, sc2_observations):
        """
        Converts a batch of numpy observations to appropriate pySC2 agent observations.
        :param np.ndarray observations: the numpy array corresponding to the observations.
        :param np.ndarray rewards: the numpy array corresponding to the rewards.
        :param list[TimeStep] sc2_observations: the original or approximate observations in pySC2 features form.
        :rtype: list[TimeStep]
        :return: a list containing the converted agent observations.
        """
        pass

    @abstractmethod
    def to_images(self, sc2_observations, features=None):
        """
        Converts the `feature_screen` layers of a batch of pySC2 observations into image representations.
        :param list[TimeStep] sc2_observations: the observations in pySC2 features form.
        :param list[str] features: the list of screen features to be rendered into the image.
        If set to `None`, all available layers are used.
        :rtype: list[Image.Image]
        :return: a list containing the image renderings for each of the given observations.
        """
        pass

    @property
    @abstractmethod
    def observation_dim(self):
        """
        Gets the size of the observation arrays.
        :rtype: int
        :return: the size of the observations.
        """
        pass

    @property
    @abstractmethod
    def action_dim(self):
        """
        Gets the size of the action arrays.
        :rtype: int
        :return: the size of the actions.
        """
        pass
