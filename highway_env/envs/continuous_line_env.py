from __future__ import division, print_function, absolute_import
import numpy as np

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle
from highway_env.vehicle.dynamics import Obstacle


class ContinuousLineEnv(AbstractEnv):
    """
        A highway merge negotiation environment.

        The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
        It is rewarded for maintaining a high velocity and avoiding collisions, but also making room for merging
        vehicles.
    """

    COLLISION_REWARD = 0
    LEFT_LANE_CONSTRAINT = 1
    LEFT_LANE_REWARD = 0.2
    HIGH_VELOCITY_REWARD = 0.8

    KIN_OBS_FEATURES = ['x', 'y', 'vx', 'vy']
    KIN_OBS_VEHICLES = 6

    DEFAULT_CONFIG = {"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                      "centering_position": [0.3, 0.5]}

    def __init__(self):
        super(ContinuousLineEnv, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        self.observation_type = "time_to_collision"
        self.steps = 0
        self.reset()

    def configure(self, config):
        self.config.update(config)

    def step(self, action):
        self.steps += 1
        return super(ContinuousLineEnv, self).step(action)

    def _observation(self):
        return super(ContinuousLineEnv, self)._observation()

    def _reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity on lanes to the right and avoiding collisions, but
            an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low velocity.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)

        reward = self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1) \
            + self.LEFT_LANE_REWARD * (len(neighbours) - 1 - self.vehicle.target_lane_index[2]) / (len(neighbours) - 1)
        lambda_ = 0
        reward -= lambda_ * self._constraint(action)
        if lambda_ > 1:
            reward /= lambda_
        # return utils.remap(reward,
        #                    [-lambda_, self.HIGH_VELOCITY_REWARD],
        #
        return reward

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        return self.vehicle.crashed

    def _constraint(self, action):
        """
            The constraint signal is the occurrence of collision
        """
        return float(self.vehicle.crashed) + float(self.vehicle.lane_index[2] == 0)/15

    def reset(self):
        self.steps = 0
        self._make_road()
        self._make_vehicles()
        return self._observation()

    def _make_road(self):
        """
            Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        length = 800
        net.add_lane("a", "b", StraightLane([0, 0], [length, 0],
                                            line_types=[LineType.CONTINUOUS_LINE, LineType.STRIPED]))
        net.add_lane("a", "b", StraightLane([0, StraightLane.DEFAULT_WIDTH], [length, StraightLane.DEFAULT_WIDTH],
                                            line_types=[LineType.NONE, LineType.CONTINUOUS_LINE]))
        net.add_lane("b", "a", StraightLane([length, 0], [0, 0],
                                            line_types=[LineType.NONE, LineType.NONE]))

        road = Road(network=net, np_random=self.np_random)
        self.road = road

    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = MDPVehicle(road, road.network.get_lane(("a", "b", 1)).position(30, 0), velocity=30)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(3):
            self.road.vehicles.append(
                vehicles_type(road,
                              position=road.network.get_lane(("a", "b", 1))
                              .position(70+40*i + 10*self.np_random.randn(), 0),
                              heading=road.network.get_lane(("a", "b", 1)).heading_at(70+40*i),
                              velocity=24 + 2*self.np_random.randn(),
                              enable_lane_change=False)
            )
        for i in range(2):
            v = vehicles_type(road,
                              position=road.network.get_lane(("b", "a", 0))
                              .position(200+100*i + 10*self.np_random.randn(), 0),
                              heading=road.network.get_lane(("b", "a", 0)).heading_at(200+100*i),
                              velocity=20 + 5*self.np_random.randn(),
                              enable_lane_change=False)
            v.target_lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)

