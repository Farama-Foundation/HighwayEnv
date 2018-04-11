from __future__ import division, print_function

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
    metadata = {'render.modes': ['human']}

    COLLISION_COST = 10
    LANE_CHANGE_COST = 0.0
    RIGHT_LANE_REWARD = 0.4
    HIGH_VELOCITY_REWARD = 1.0
    EPISODE_SUCCESS = 10.0

    def __init__(self):
        road = Road.create_random_road(lanes_count=4, lane_width=4.0, vehicles_count=20, vehicles_type=IDMVehicle)
        vehicle = MDPVehicle.create_random(road, 25)
        road.vehicles.append(vehicle)
        super(HighwayEnv, self).__init__(road, vehicle)

    def observation(self):
        return 1

    def reward(self, action):
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        action_reward = {0: -self.LANE_CHANGE_COST, 1: 0, 2: -self.LANE_CHANGE_COST, 3: 0, 4: 0}
        state_reward = \
            - self.COLLISION_COST * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * self.vehicle.lane_index \
            + self.HIGH_VELOCITY_REWARD * self.vehicle.speed_index() \
            + self.EPISODE_SUCCESS * self.all_vehicles_passed()
        return action_reward[action] + state_reward

    def is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or if it successfully passed all other vehicles.
        """
        return self.vehicle.crashed or self.all_vehicles_passed()

    def reset(self):
        # TODO
        pass

    def all_vehicles_passed(self):
        return len(self.road.vehicles) > 1 and (self.vehicle.position[0] > 50 + max(
            [o.position[0] for o in self.road.vehicles if o is not self.vehicle]))