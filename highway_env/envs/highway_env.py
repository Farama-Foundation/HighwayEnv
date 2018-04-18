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
    LEFT_LANE_REWARD = -0.1
    HIGH_VELOCITY_REWARD = 0.2
    EPISODE_SUCCESS_REWARD = 1.0
    LANE_CHANGE_REWARD = -0

    MAXIMUM_SIMULATION_STEPS = 3 * 60

    def __init__(self):
        super(HighwayEnv, self).__init__()
        self.road, self.vehicle = self._create_road()

    def reset(self):
        self.road, self.vehicle = self._create_road()
        return super(HighwayEnv, self).reset()

    def _create_road(self):
        road = Road.create_random_road(lanes_count=4,
                                       vehicles_count=5,
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
            + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1) \
            + self.EPISODE_SUCCESS_REWARD * self._all_vehicles_passed()
        return action_reward[action] + state_reward

    def _observation(self):
        return self._simplified()

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or if it successfully passed all other vehicles.
        """
        return self.vehicle.crashed or self._all_vehicles_passed() or self.steps > self.MAXIMUM_SIMULATION_STEPS

    def _all_vehicles_passed(self):
        return len(self.road.vehicles) > 1 and (self.vehicle.position[0] > 50 + max(
            [o.position[0] for o in self.road.vehicles if o is not self.vehicle]))