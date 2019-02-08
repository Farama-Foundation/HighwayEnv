from __future__ import division, print_function, absolute_import
import numpy as np
import pandas
from gym import GoalEnv, spaces

from highway_env.envs.abstract import AbstractEnv
from highway_env.envs.graphics import EnvViewer
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.dynamics import Vehicle, Obstacle


class ParkingEnv(AbstractEnv, GoalEnv):
    """
        A continuous control environment.

        It implements a reach-type task, where the agent observes their position and velocity and must
        control their acceleration and steering so as to reach a given goal.

        Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    STEERING_RANGE = np.pi / 4
    ACCELERATION_RANGE = 5.0

    OBS_SCALE = 100
    REWARD_WEIGHTS = [1 / 100, 1 / 100, 1 / 100, 1 / 100, 1 / 10, 1/10]
    SUCCESS_GOAL_REWARD = 0.15

    DEFAULT_CONFIG = {
        "centering_position": [0.5, 0.5]
    }

    KIN_OBS_FEATURES = ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h']
    KIN_OBS_VEHICLES = 1
    NORMALIZE_OBS = False

    def __init__(self):
        super(ParkingEnv, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
            observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
        ))
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.REWARD_WEIGHTS = np.array(self.REWARD_WEIGHTS)
        EnvViewer.SCREEN_HEIGHT = EnvViewer.SCREEN_WIDTH // 2

    def step(self, action):
        # Forward action to the vehicle
        self.vehicle.act({
            "acceleration": action[0] * self.ACCELERATION_RANGE,
            "steering": action[1] * self.STEERING_RANGE
        })
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

    def _create_road(self, spots=15):
        """
            Create a road composed of straight adjacent lanes.
        """
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 8
        for k in range(spots):
            x = (k - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))

        self.road = Road(network=net,
                         np_random=self.np_random)

    def _create_vehicles(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        self.vehicle = Vehicle(self.road, [0, 0], 2*np.pi*self.np_random.rand(), 0)
        self.road.vehicles.append(self.vehicle)

        lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = Obstacle(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
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

    def compute_reward(self, achieved_goal, desired_goal, info, p=0.5):
        """
            Proximity to the goal is rewarded

            We use a weighted p-norm
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return - np.power(np.dot(self.OBS_SCALE * np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p)

    def _reward(self, action):
        raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        return self.compute_reward(achieved_goal, desired_goal, None) > -self.SUCCESS_GOAL_REWARD

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the goal is reached.
        """
        obs = self._observation()
        return self.vehicle.crashed  # or self._is_success(obs['achieved_goal'], obs['desired_goal'])
