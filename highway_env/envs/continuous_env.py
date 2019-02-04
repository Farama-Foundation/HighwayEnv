from __future__ import division, print_function, absolute_import
import numpy as np
import pandas
from gym import GoalEnv, spaces

from highway_env.envs.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.dynamics import Vehicle, Obstacle


class ContinuousEnv(AbstractEnv, GoalEnv):
    """
        A continuous control environment.

        It implements a reach-type task, where the agent observes their position and velocity and must
        control their acceleration and steering so as to reach a given goal.

        Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    STEERING_RANGE = np.pi / 4
    ACCELERATION_RANGE = 5.0

    OBS_SCALE = 100
    REWARD_SCALE = 100
    SUCCESS_GOAL_DISTANCE = 5

    DEFAULT_CONFIG = {
        "centering_position": [0.5, 0.5]
    }

    OBSERVATION_FEATURES = ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h']
    OBSERVATION_VEHICLES = 1
    NORMALIZE_OBS = False

    def __init__(self):
        super(ContinuousEnv, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        # self._max_episode_steps = 50
        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
            observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
        ))
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)

    def step(self, action):
        # Forward action to the vehicle
        self.vehicle.act({"steering": action[0] * self.STEERING_RANGE,
                          "acceleration": action[1] * self.ACCELERATION_RANGE})
        self._simulate()

        obs = self._observation()
        info = {"is_success": self._is_success(obs['achieved_goal'], obs['desired_goal'])}
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        terminal = self._is_terminal()

        return obs, reward, terminal, info

    def reset(self):
        self._create_road()
        self._create_vehicles()
        return self._observation()

    def configure(self, config):
        self.config.update(config)

    def _create_road(self):
        """
            Create a road composed of straight adjacent lanes.
        """
        self.road = Road(network=RoadNetwork.straight_road_network(lanes=4),
                         np_random=self.np_random)

    def _create_vehicles(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        self.vehicle = Vehicle(self.road, [200, self.np_random.randint(0, 12)], 2*np.pi*self.np_random.rand(), 0)
        self.road.vehicles.append(self.vehicle)
        self.goal = Obstacle(self.road, [self.vehicle.position[0] + self.np_random.randint(-100, 100),
                                         self.np_random.randint(0, 12)])
        self.goal.COLLISIONS_ENABLED = False
        self.road.vehicles.insert(0, self.goal)

    def _observation(self):
        obs = np.ravel(pandas.DataFrame.from_records([self.vehicle.to_dict()])[self.OBSERVATION_FEATURES])
        goal = np.ravel(pandas.DataFrame.from_records([self.goal.to_dict()])[self.OBSERVATION_FEATURES])
        obs = {
            "observation": obs / self.OBS_SCALE,
            "achieved_goal": obs / self.OBS_SCALE,
            "desired_goal": goal / self.OBS_SCALE
        }
        return obs

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
            Proximity to the goal is rewarded
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param info: any supplementary information
        :return: the corresponding reward
        """
        return -np.linalg.norm(achieved_goal - desired_goal, axis=-1) * self.OBS_SCALE / self.REWARD_SCALE

    def _reward(self, action):
        raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        return np.linalg.norm(achieved_goal - desired_goal) * self.OBS_SCALE < self.SUCCESS_GOAL_DISTANCE

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the goal is reached.
        """
        obs = self._observation()
        return self.vehicle.crashed  # or self._is_success(obs['achieved_goal'], obs['desired_goal'])
