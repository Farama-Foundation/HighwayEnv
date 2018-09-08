from __future__ import division, print_function, absolute_import
import numpy as np

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.envs.graphics import EnvViewer
from highway_env.road.lane import LineType, StraightLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import MDPVehicle


class RoundaboutEnv(AbstractEnv):

    COLLISION_REWARD = -1
    HIGH_VELOCITY_REWARD = 0.2
    RIGHT_LANE_REWARD = 0
    LANE_CHANGE_REWARD = 0

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
        center = [0, -length]
        radius = [length, length+4]
        line = [[LineType.CONTINUOUS, LineType.STRIPED], [LineType.NONE, LineType.CONTINUOUS]]
        alpha = 10

        net = RoadNetwork()
        for lane in [0, 1]:
            net.add_lane(0, 1, CircularLane(center, radius[lane], rad(90-alpha), rad(alpha), line_types=line[lane]))
            net.add_lane(1, 2, CircularLane(center, radius[lane], rad(alpha), rad(-alpha), line_types=line[lane]))
            net.add_lane(2, 3, CircularLane(center, radius[lane], rad(-alpha), rad(-90+alpha), line_types=line[lane]))
            net.add_lane(3, 4, CircularLane(center, radius[lane], rad(-90+alpha), rad(-90-alpha), line_types=line[lane]))
            net.add_lane(4, 5, CircularLane(center, radius[lane], rad(-90-alpha), rad(-180+alpha), line_types=line[lane]))
            net.add_lane(5, 6, CircularLane(center, radius[lane], rad(-180+alpha), rad(-180-alpha), line_types=line[lane]))
            net.add_lane(6, 7, CircularLane(center, radius[lane], rad(180-alpha), rad(90+alpha), line_types=line[lane]))
            net.add_lane(7, 0, CircularLane(center, radius[lane], rad(90+alpha), rad(90-alpha), line_types=line[lane]))
        net.add_lane(10, 0, StraightLane([0, 50], [10, 6]))

        road = Road(network=net)
        self.road = road

    def make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = MDPVehicle(road,
                                 road.network.get_lane((10, 0, 0)).position(0, 0),
                                 velocity=10,
                                 heading=road.network.get_lane((10, 0, 0)).heading)
        MDPVehicle.SPEED_MIN = 5
        MDPVehicle.SPEED_MAX = 20
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(3):
            road.vehicles.append(other_vehicles_type(road,
                                                     road.network.get_lane((6, 7, 0)).position(-10*i, 0),
                                                     velocity=10))


def rad(deg):
    return deg*np.pi/180
