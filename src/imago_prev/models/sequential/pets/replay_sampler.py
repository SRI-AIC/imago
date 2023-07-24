import logging
import multiprocessing
from pysc2.env.environment import TimeStep
from pysc2.lib.actions import FunctionCall
from pysc2.lib.features import AgentInterfaceFormat
from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2recorder.replayer import DebugReplayProcessor, DebugStepListener

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

PLAYER_PERSPECTIVE = 1


class ReplaySamplerProcessor(DebugReplayProcessor):
    def __init__(self, aif, samples_queue, enqueue_frequency, step_mul=8):
        """
        Creates a new replay sampler processor.
        :param AgentInterfaceFormat aif: the pysc2 agent interface format.
        :param multiprocessing.Queue samples_queue: the queue to put samples in.
        :param int enqueue_frequency: the frequency at which to enqueue the sample data.
        :param int step_mul: the step multiplier for the SC2 environment.
        """
        self._aif = aif
        self._step_mul = step_mul
        self._listener = ReplaySamplerListener(samples_queue, enqueue_frequency)

    @property
    def step_mul(self):
        return self._step_mul

    @property
    def interface(self):
        interface = sc_pb.InterfaceOptions(
            raw=True, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=self._aif.camera_width_world_units))
        self._aif.feature_dimensions.screen.assign_to(interface.feature_layer.resolution)
        self._aif.feature_dimensions.minimap.assign_to(interface.feature_layer.minimap_resolution)
        return interface

    @property
    def agent_interface_format(self):
        return self._aif

    def create_listeners(self):
        return [self._listener]


class ReplaySamplerListener(DebugStepListener):

    def __init__(self, samples_queue, enqueue_frequency):
        """
        Creates a new replay sampler listener.
        :param multiprocessing.Queue samples_queue: the queue to put samples in.
        :param int enqueue_frequency: the frequency at which to enqueue the sample data.
        """
        self._samples_queue = samples_queue
        self._enqueue_frequency = enqueue_frequency

        self._ignore_replay = False
        self._replay_name = ''
        self._total_steps = 0
        self._cur_ep = 0
        self._total_eps = 0

        self._pb_obs_buffer = []
        self._agent_obs_buffer = []
        self._agent_actions_buffer = []

    def start_replay(self, replay_name, replay_info, player_perspective):
        """
        Called when starting a new replay.
        :param str replay_name: replay file name.
        :param ResponseReplayInfo replay_info: protobuf message.
        :param int player_perspective: ID of player whose perspective we see observations.
        :return:
        """
        # ignore if not player's side
        self._ignore_replay = player_perspective != PLAYER_PERSPECTIVE

        if not self._ignore_replay:
            self._replay_name = replay_name
            self._total_steps = 0
            logging.info('Collecting data from replay \'{}\'...'.format(replay_name))

    def finish_replay(self):
        """
        Saves the features history to a CSV file.
        """
        if not self._ignore_replay:
            logging.info('Collected {} samples from {} episodes from replay \'{}\'...'.format(
                self._total_steps, self._total_eps, self._replay_name))
            self._put_samples()

    def reset(self, pb_obs, agent_obs):
        """
        Called for the first observation of the replay. Records the current observation.
        :param ResponseObservation pb_obs: the observation in protobuf form.
        :param TimeStep agent_obs: the observation in pysc2 features form.
        :return:
        """
        if not self._ignore_replay:
            self._pb_obs_buffer.append(pb_obs)
            self._agent_obs_buffer.append(agent_obs)
            self._agent_actions_buffer.append([])

    def step(self, ep, step, pb_obs, agent_obs, agent_actions):
        """
        Updates the feature extractor with the given observations.
        :param int ep: the episode that this observation was made.
        :param int step: the episode time-step in which this observation was made.
        :param ResponseObservation pb_obs: the observation in protobuf form.
        :param TimeStep agent_obs: the observation in pysc2 features form.
        :param list[FunctionCall] agent_actions: list of actions executed by the agent between the previous observation
        and the current observation.
        :return:
        """
        if self._ignore_replay:
            return

        if ep != self._cur_ep:
            self._cur_ep = ep
            self._total_eps += 1

        # update lists
        self._pb_obs_buffer.append(pb_obs)
        self._agent_obs_buffer.append(agent_obs)
        self._agent_actions_buffer.append(agent_actions)

        # checks whether to put data in queue
        self._total_steps += 1
        if self._total_steps % self._enqueue_frequency == 0:
            self._put_samples()

    def _put_samples(self):
        if len(self._pb_obs_buffer) == 0:
            return
        self._samples_queue.put((self._pb_obs_buffer, self._agent_obs_buffer, self._agent_actions_buffer))
        self._pb_obs_buffer = []
        self._agent_obs_buffer = []
        self._agent_actions_buffer = []
