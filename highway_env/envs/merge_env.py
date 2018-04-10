from __future__ import division, print_function
import numpy as np

from highway_env.envs.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, LanesConcatenation
from highway_env.road.road import Road
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle
from highway_env.vehicle.dynamics import Obstacle


class MergeEnv(AbstractEnv):
    """
        A highway merge negotiation environment.

        The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
        It is rewarded for maintaining a high velocity and avoiding collisions, but also making room for merging
        vehicles.
    """

    VELOCITY_REWARD = 1.0
    MERGING_VELOCITY_REWARD = 2.0 / 20.0
    RIGHT_LANE_REWARD = 0.5
    ACCELERATION_COST = 0
    LANE_CHANGE_COST = 0
    COLLISION_COST = 10

    def __init__(self):
        road = MergeEnv.make_road()
        vehicle = MergeEnv.make_vehicles(road)
        super(MergeEnv, self).__init__(road, vehicle)

    def observation(self):
        return 1

    def reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity on lanes to the right and avoiding collisions, but
            an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low velocity.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: -self.LANE_CHANGE_COST,
                         1: 0,
                         2: -self.LANE_CHANGE_COST,
                         3: -self.ACCELERATION_COST,
                         4: -self.ACCELERATION_COST}
        reward = -self.COLLISION_COST * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * self.vehicle.lane_index \
            + self.VELOCITY_REWARD * self.vehicle.velocity_index

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == len(self.road.lanes)-1 and isinstance(vehicle, ControlledVehicle):
                reward -= self.MERGING_VELOCITY_REWARD * (vehicle.target_velocity - vehicle.velocity)
        return reward + action_reward[action]

    def is_terminal(self):
        return self.vehicle.crashed or self.vehicle.position[0] > 400

    def reset(self):
        # TODO
        pass

    @staticmethod
    def make_road():
        """
            Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        ends = [80, 80, 80]
        l0 = StraightLane(np.array([0, 0]), 0, 4.0, [LineType.CONTINUOUS, LineType.NONE])
        l1 = StraightLane(np.array([0, 4]), 0, 4.0, [LineType.STRIPED, LineType.CONTINUOUS])

        lc0 = StraightLane(np.array([0, 6.5 + 4 + 4]), 0, 4.0,
                           [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[-np.inf, ends[0]], forbidden=True)
        amplitude = 3.3
        lc1 = SineLane(lc0.position(ends[0], -amplitude), 0, 4.0, amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2,
                       [LineType.STRIPED, LineType.STRIPED], bounds=[0, ends[1]], forbidden=True)
        lc2 = StraightLane(lc1.position(ends[1], 0), 0, 4.0,
                           [LineType.NONE, LineType.CONTINUOUS], bounds=[0, ends[2]], forbidden=True)
        l2 = LanesConcatenation([lc0, lc1, lc2])
        road = Road([l0, l1, l2])
        road.vehicles.append(Obstacle(road, lc2.position(ends[2], 0)))
        return road

    @staticmethod
    def make_vehicles(road):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :param road: the road on which the vehicles drive
        :return: the ego-vehicle
        """
        ego_vehicle = MDPVehicle(road, road.lanes[-2].position(-40, 0), velocity=30)
        road.vehicles.append(ego_vehicle)

        road.vehicles.append(IDMVehicle(road, road.lanes[0].position(20, 0), velocity=30))
        road.vehicles.append(IDMVehicle(road, road.lanes[1].position(35, 0), velocity=30))
        road.vehicles.append(IDMVehicle(road, road.lanes[0].position(-65, 0), velocity=31.5))

        merging_v = IDMVehicle(road, road.lanes[-1].position(70, 0), velocity=20)
        merging_v.TIME_WANTED = 1.0
        merging_v.POLITENESS = 0.0
        merging_v.target_velocity = 30
        road.vehicles.append(merging_v)
        return ego_vehicle
