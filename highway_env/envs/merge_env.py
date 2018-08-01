from __future__ import division, print_function, absolute_import
import numpy as np

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, LanesConcatenation
from highway_env.road.road import Road
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle
from highway_env.vehicle.dynamics import Obstacle


class MergeEnv(AbstractEnv):
    """
        A highway merge negotiation environment.

        The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
        It is rewarded for maintaining a high velocity and avoiding collisions, but also making room for merging
        vehicles.
    """

    COLLISION_REWARD = -1
    RIGHT_LANE_REWARD = 0.1
    HIGH_VELOCITY_REWARD = 0.2
    MERGING_VELOCITY_REWARD = -0.5
    LANE_CHANGE_REWARD = -0.05

    DEFAULT_CONFIG = {"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle"}

    def __init__(self):
        super(MergeEnv, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        self.make_road()
        self.make_vehicles()

    def configure(self, config):
        self.config.update(config)

    def _observation(self):
        return super(MergeEnv, self)._observation()

    def _reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity on lanes to the right and avoiding collisions, but
            an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low velocity.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.LANE_CHANGE_REWARD,
                         1: 0,
                         2: self.LANE_CHANGE_REWARD,
                         3: 0,
                         4: 0}
        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.RIGHT_LANE_REWARD * self.vehicle.lane_index / (len(self.road.lanes) - 2) \
                 + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == len(self.road.lanes)-1 and isinstance(vehicle, ControlledVehicle):
                reward += self.MERGING_VELOCITY_REWARD * \
                          (vehicle.target_velocity - vehicle.velocity) / vehicle.target_velocity
        return reward + action_reward[action]

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        return self.vehicle.crashed or self.vehicle.position[0] > 300

    def reset(self):
        self.make_road()
        self.make_vehicles()
        return self._observation()

    def make_road(self):
        """
            Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        ends = [80, 80, 80]
        l0 = StraightLane(np.array([0, 0]), 0, 4.0, [LineType.CONTINUOUS_LINE, LineType.NONE])
        lm0 = StraightLane(np.array([0, 4]), 0, 4.0,
                           [LineType.STRIPED, LineType.CONTINUOUS_LINE], bounds=[-np.inf, sum(ends[0:2])])
        lm1 = StraightLane(lm0.position(sum(ends[0:2]), 0), 0, 4.0,
                           [LineType.STRIPED, LineType.STRIPED], bounds=[0, ends[2]])
        lm2 = StraightLane(lm1.position(ends[2], 0), 0, 4.0,
                           [LineType.STRIPED, LineType.CONTINUOUS_LINE], bounds=[0, np.inf])
        l1 = LanesConcatenation([lm0, lm1, lm2])

        lc0 = StraightLane(np.array([0, 6.5 + 4 + 4]), 0, 4.0,
                           [LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE], bounds=[-np.inf, ends[0]], forbidden=True)
        amplitude = 3.25
        lc1 = SineLane(lc0.position(ends[0], -amplitude), 0, 4.0, amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2,
                       [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[0, ends[1]], forbidden=True)
        lc2 = StraightLane(lc1.position(ends[1], 0), 0, 4.0,
                           [LineType.NONE, LineType.CONTINUOUS_LINE], bounds=[0, ends[2]], forbidden=True)
        l2 = LanesConcatenation([lc0, lc1, lc2])
        road = Road([l0, l1, l2])
        road.vehicles.append(Obstacle(road, lc2.position(ends[2], 0)))
        self.road = road

    def make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = MDPVehicle(road, road.lanes[-2].position(-40, 0), velocity=30)
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        road.vehicles.append(other_vehicles_type(road, road.lanes[0].position(20, 0), velocity=29))
        road.vehicles.append(other_vehicles_type(road, road.lanes[1].position(0, 0), velocity=31))
        road.vehicles.append(other_vehicles_type(road, road.lanes[0].position(-65, 0), velocity=31.5))

        merging_v = other_vehicles_type(road, road.lanes[-1].position(40, 0), velocity=20)
        merging_v.target_velocity = 30
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle
