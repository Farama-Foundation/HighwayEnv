from __future__ import division, print_function, absolute_import
import numpy as np
from gym import logger

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.road import Road
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.control import MDPVehicle


class HighwayEnv(AbstractEnv):
    """
        A highway driving environment.

        The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high velocity,
        staying on the rightmost lanes and avoiding collisions.
    """

    COLLISION_REWARD = -1
    """ The reward received when colliding with a vehicle."""
    RIGHT_LANE_REWARD = 0.1
    """ The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""
    HIGH_VELOCITY_REWARD = 0.2
    """ The reward received when driving at full speed, linearly mapped to zero for lower speeds."""
    LANE_CHANGE_REWARD = -0
    """ The reward received at each lane change action."""

    DIFFICULTY_LEVELS = {
        "EASY": {
            "lanes_count": 2,
            "initial_spacing": 2,
            "vehicles_count": 5,
            "duration": 20,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle"
        },
        "MEDIUM": {
            "lanes_count": 3,
            "initial_spacing": 2,
            "vehicles_count": 10,
            "duration": 30,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle"
        },
        "HARD": {
            "lanes_count": 4,
            "initial_spacing": 3,
            "vehicles_count": 15,
            "duration": 40,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle"
        },
    }

    def __init__(self):
        super(HighwayEnv, self).__init__()
        self.config = self.DIFFICULTY_LEVELS["HARD"].copy()
        self.steps = 0
        self.reset()

    def reset(self):
        self.road, self.vehicle = self._create_road()
        self.steps = 0
        return self._observation()

    def step(self, action):
        self.steps += 1
        return super(HighwayEnv, self).step(action)

    def set_difficulty_level(self, level):
        if level in self.DIFFICULTY_LEVELS:
            logger.info("Set difficulty level to: {}".format(level))
            self.config.update(self.DIFFICULTY_LEVELS[level])
            self.reset()
        else:
            raise ValueError("Invalid difficulty level, choose among {}".format(str(self.DIFFICULTY_LEVELS.keys())))

    def configure(self, config):
        self.config.update(config)

    def _create_road(self):
        road = Road.create_random_road(lanes_count=self.config["lanes_count"],
                                       vehicles_count=self.config["vehicles_count"],
                                       vehicles_type=utils.class_from_path(self.config["other_vehicles_type"]),
                                       np_random=self.np_random)
        vehicle = MDPVehicle.create_random(road, 25, spacing=self.config["initial_spacing"], np_random=self.np_random)
        road.vehicles.append(vehicle)
        return road, vehicle

    def _reward(self, action):
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        action_reward = {0: self.LANE_CHANGE_REWARD, 1: 0, 2: self.LANE_CHANGE_REWARD, 3: 0, 4: 0}
        state_reward = \
            + self.COLLISION_REWARD * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * self.vehicle.target_lane_index / (len(self.road.lanes) - 1) \
            + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)
        return action_reward[action] + state_reward

    def _observation(self):
        return super(HighwayEnv, self)._observation()

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        return self.vehicle.crashed or self.steps > self.config["duration"]
