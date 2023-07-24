""" This implements an Agent that wraps another Agent, similar to recorder.py.  
However, this attaches extra recording code into the step function.
"""

import os

import numpy as np

from absl import app
from absl import flags

from PIL import Image
import time
import threading
import importlib

from pdb import set_trace

import pysc2
from pysc2.agents.base_agent import BaseAgent
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import actions, features, point
from pysc2.lib import point_flag

from .featurizers import compute_unit_type_hist
import json

from pysc2.lib import features
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
point_flag.DEFINE_point("feature_screen_size", "64",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
#point_flag.DEFINE_point("rgb_screen_size", "256",
point_flag.DEFINE_point("rgb_screen_size", 256,                        
                        "Resolution for rendered screen.")
# point_flag.DEFINE_point("rgb_minimap_size", "256",
point_flag.DEFINE_point("rgb_minimap_size", 256,
                        "Resolution for rendered minimap.")
flags.DEFINE_integer("stride", 1, "Stride for scanning the map.")

flags.DEFINE_enum("action_space", "RGB", sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("use_raw_units", True,
                  "Whether to include raw units.")
flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 1, "Total episodes.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_enum("agent_race", "terran", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Wrapped agent's race.")
flags.DEFINE_string("agent", "Bot", "First agent (wrapped), either Bot or agent class.")

flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_string("agent2_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, its strength.")
flags.DEFINE_enum("bot_build", "random", sc2_env.BotBuild._member_names_,  # pylint: disable=protected-access
                  "Bot's build strategy.")

flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.DEFINE_bool("battle_net_map", False, "Use the battle.net map version.")
flags.mark_flag_as_required("map")

flags.DEFINE_string("output_dir", "./output", "Root directory for output")
flags.mark_flag_as_required("output_dir")

flags.DEFINE_list("features",
                  "rgb_screen,feature_screen",
                  "Names of features to record.")


class ScanFinished(RuntimeError):
    pass


class NopAgent(BaseAgent):
    def __init__(self, real_time=False):
        super().__init__()
        self.real_time = real_time

    def step(self, obs_batch):
        if self.real_time:
            time.sleep(1 / 22.4)
        return super().step(obs_batch)


# ----------------------------------------------------------------------------

class WrapObserverAgent(BaseAgent):
    """
    Takes an existing agent and wraps a recorder around it that saves the
    features out.
    """
    def __init__(self, wrapped_agent, base_dir, env, stride=1, feature_names=None):
        super().__init__()
        self.wrapped_agent = wrapped_agent
        self.base_dir = base_dir
        self.stride = stride

        if feature_names is None:
            feature_names = [
                "rgb_screen", "rgb_minimap",
                "feature_screen", "feature_minimap",
                "available_actions", "player"]
        self.feature_names = feature_names

        self._aif = env._interface_formats[0]
        if self._aif.rgb_dimensions.screen < self._aif.feature_dimensions.screen:
            raise ValueError("Screen RGB dimension must be >= feature dimension")
        if self._aif.rgb_dimensions.minimap < self._aif.feature_dimensions.minimap:
            raise ValueError("Minimap RGB dimension must be >= feature dimension")

        self._game_info = env._game_info[0]
        self._map_size = point.Point.build(self._game_info.start_raw.map_size)
        self._playable = point.Rect(
            point.Point.build(self._game_info.start_raw.playable_area.p0),
            point.Point.build(self._game_info.start_raw.playable_area.p1))
        print("map_size: {}".format(self._map_size))
        print("playable: {}".format(self._playable))

        self.episode_step = 0  # Episode specific stgep

        # self._minimap_dim = self._aif.rgb_dimensions.minimap

        # self._cam_x = None
        # self._cam_y = None
        #
        # self._x = 0
        # self._y = 0
        # self._cam_x = 0
        # self._cam_y = 0
        #
        # self._state = "init"
        # self._xmin = None
        # self._xmax = None
        # self._ymin = None
        # self._ymax = None

    def _capture(self, obs):
        """
        TODO: Consider if this can be optimized
        """        
        features = {"steps": int(self.steps), "obs_reward": float(obs.reward), "total_reward": float(self.reward) }
        # compute_unit_type_hist(obs, features)
        #print("[{}]: reward={}\tcamera x, y: {}, {}".format(self.steps, self.reward, self._x, self._y))
        idx = "e{}_s{}".format(self.episodes, self.steps)
        # Dump out features
        with open(os.path.join(self.base_dir, "{}.features.json".format(idx)), "w") as f:
            json.dump(features, f)
        # NOTE: feature_screen.unit_type is a width x height array.  We will not get unit affiliations from this.
        # We will need to pull this from obs.observation.feature_screen.unit_density (see 
        # https://github.com/deepmind/pysc2/blob/master/docs/environment.md#observation)
        unit_type_screen = obs.observation.feature_screen.unit_type
        np.save(os.path.join(self.base_dir, "featurescreen_unit_type_{}.npy".format(idx)), unit_type_screen)
        for f in self.feature_names:
            x = obs.observation[f]
            if f in ["rgb_screen", "rgb_minimap"]:
                img = Image.fromarray(x.astype(np.uint8), mode="RGB")
                img.save(os.path.join(self.base_dir, "{}_{}.png".format(f, idx)))

    def finalize(self):
        """
        Resets the
        :return:
        """

    def reset(self):
        """ Called at the beginning of episode."""
        super(WrapObserverAgent, self).reset()
        self.episode_step = 0

    def step(self, obs):
        self._capture(obs)
        self.steps += 1
        self.episode_step += 1
        self.reward += obs.reward

        wrapped_action = self.wrapped_agent.step(obs)
        return wrapped_action


# ----------------------------------------------------------------------------

def run_thread(wrapped_agent_cls, agents, players, map_name, agent_interface_format, visualize):
    """Run one thread worth of the environment with agents."""
    with sc2_env.SC2Env(
            map_name=map_name,
            battle_net_map=FLAGS.battle_net_map,
            players=players,
            agent_interface_format=agent_interface_format,
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            disable_fog=FLAGS.disable_fog,
            visualize=visualize) as env:
        wrapped_agent = wrapped_agent_cls()
        observer = WrapObserverAgent(
            wrapped_agent,
            FLAGS.output_dir, env, stride=FLAGS.stride, feature_names=FLAGS.features)
        agents = [observer] + agents
        env = available_actions_printer.AvailableActionsPrinter(env)
        try:
            run_loop.run_loop(agents, env, FLAGS.max_agent_steps, FLAGS.max_episodes)
        except ScanFinished:
            return
        # if FLAGS.save_replay:
        # env.save_replay(agent_classes[0].__name__)


def main(unused_argv):
    """Run an observer that is wrapped around a default agent.  Invocation is similar to invoking pysc2.bin.agent.
    Output directory is under output_dir, with the default under "output/".
    Observations are now indexed by episode number and step.
    """

    map_inst = maps.get(FLAGS.map)
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    agent_interface_format = sc2_env.parse_agent_interface_format(
        feature_screen=FLAGS.feature_screen_size,
        feature_minimap=FLAGS.feature_minimap_size,
        rgb_screen=FLAGS.rgb_screen_size,
        rgb_minimap=FLAGS.rgb_minimap_size,
        action_space=FLAGS.action_space,
        use_feature_units=FLAGS.use_feature_units,
        use_raw_units=FLAGS.use_raw_units,
        use_camera_position=True)

    agents = []
    players = []

    # Use the "agent" argument to identify the agent to wrap an observer around.
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    wrapped_agent_cls = getattr(importlib.import_module(agent_module), agent_name)

    players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race], "WrapObserverAgent"))

    if map_inst.players >= 2:
        if FLAGS.agent2 == "Bot":
            players.append(sc2_env.Bot(sc2_env.Race[FLAGS.agent2_race],
                                       sc2_env.Difficulty[FLAGS.difficulty],
                                       sc2_env.BotBuild[FLAGS.bot_build]))
        else:
            agents.append(NopAgent())
            players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent2_race], "dummy"))

    threads = []
    for _ in range(FLAGS.parallel - 1):
        t = threading.Thread(target=run_thread,
                             args=(agents, players, FLAGS.map,
                                   agent_interface_format, False))
        threads.append(t)
        t.start()

    run_thread(wrapped_agent_cls, agents, players, FLAGS.map, agent_interface_format, FLAGS.render)

    for t in threads:
        t.join()


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
