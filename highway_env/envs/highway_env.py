from __future__ import division, print_function, absolute_import
import numpy as np

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
    LEFT_LANE_REWARD = -0.1
    """ The reward received when driving on the left-most lanes, linearly mapped to zero for other lanes."""
    HIGH_VELOCITY_REWARD = 0.2
    """ The reward received when driving at full speed, linearly mapped to zero for lower speeds."""
    LANE_CHANGE_REWARD = -0
    """ The reward received at each lane change action."""

    DURATION = 60
    """ Number of steps until the termination of the episode [s]."""

    def __init__(self):
        super(HighwayEnv, self).__init__()
        self.steps = 0
        self.reset()

    def reset(self):
        self.road, self.vehicle = self._create_road()
        self.steps = 0
        return self._observation()

    def step(self, action):
        self.steps += 1
        return super(HighwayEnv, self).step(action)

    def _create_road(self):
        road = Road.create_random_road(lanes_count=4,
                                       vehicles_count=20,
                                       vehicles_type=IDMVehicle,
                                       np_random=self.np_random)
        vehicle = MDPVehicle.create_random(road, 25, spacing=6, np_random=self.np_random)
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
            + self.LEFT_LANE_REWARD * (len(self.road.lanes)-1 - self.vehicle.lane_index) / (len(self.road.lanes) - 1) \
            + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)
        return action_reward[action] + state_reward

    def _observation(self):
        return self._simplified()

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        return self.vehicle.crashed or self.steps > self.DURATION
