from __future__ import division, print_function, absolute_import
import numpy as np

from highway_env.envs.abstract import AbstractEnv
from highway_env.envs.graphics import EnvViewer
from highway_env.road.lane import LineType, StraightLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import MDPVehicle


class RoundaboutEnv(AbstractEnv):

    COLLISION_REWARD = -1
    HIGH_VELOCITY_REWARD = 0.2

    DEFAULT_CONFIG = {"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle"}

    def __init__(self):
        super(RoundaboutEnv, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        self.make_road()
        self.make_vehicles()
        EnvViewer.SCREEN_HEIGHT = 600

    def configure(self, config):
        self.config.update(config)

    def _observation(self):
        return super(RoundaboutEnv, self)._observation()

    def _reward(self, action):
        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)
        return reward

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        return self.vehicle.crashed

    def reset(self):
        self.make_road()
        self.make_vehicles()
        return self._observation()

    def make_road(self):
        length = 40
        width = 4

        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        net = RoadNetwork()
        net.add_lane(0, 1, CircularLane([0, -length], length, np.pi/2, 0, line_types=[c, s]))
        net.add_lane(1, 2, CircularLane([0, -length], length, 0, -np.pi/2, line_types=[c, s]))
        net.add_lane(2, 3, CircularLane([0, -length], length, -np.pi/2, -np.pi, line_types=[c, s]))
        net.add_lane(3, 0, CircularLane([0, -length], length, np.pi, np.pi/2, line_types=[c, s]))
        net.add_lane(0, 1, CircularLane([0, -length], length+width, np.pi/2, 0, line_types=[n, c]))
        net.add_lane(1, 2, CircularLane([0, -length], length+width, 0, -np.pi/2, line_types=[n, c]))
        net.add_lane(2, 3, CircularLane([0, -length], length+width, -np.pi/2, -np.pi, line_types=[n, c]))
        net.add_lane(3, 0, CircularLane([0, -length], length+width, np.pi, np.pi/2, line_types=[n, c]))

        road = Road(network=net)
        self.road = road

    def make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = MDPVehicle(road, road.network.get_lane((0, 1, 0)).position(0, 0), velocity=10)
        MDPVehicle.SPEED_MIN = 5
        MDPVehicle.SPEED_MAX = 20
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
