from __future__ import division, print_function, absolute_import
from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane, AbstractLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import MDPVehicle


class IntersectionEnv(AbstractEnv):

    COLLISION_REWARD = -1
    HIGH_VELOCITY_REWARD = 0.2
    RIGHT_LANE_REWARD = 0
    LANE_CHANGE_REWARD = -0.05

    DURATION = 50

    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics"
        },
        "policy_frequency": 1,  # [Hz]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "incoming_vehicle_destination": None,
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.5, 0.6]
    }

    def __init__(self):
        super(IntersectionEnv, self).__init__()
        self.steps = 0
        self.reset()

    def _reward(self, action):
        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / max(self.vehicle.SPEED_COUNT - 1, 1)
        return utils.remap(reward, [self.COLLISION_REWARD+self.LANE_CHANGE_REWARD, self.HIGH_VELOCITY_REWARD], [0, 1])

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        return self.vehicle.crashed or self.steps >= self.DURATION

    def reset(self):
        self._make_road()
        self._make_vehicles()
        self.steps = 0
        return super(IntersectionEnv, self).reset()

    def step(self, action):
        self.steps += 1
        return super(IntersectionEnv, self).step(action)

    def _make_road(self):
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width+5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 20  # [m]

        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = rad(90 * corner)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner), StraightLane(start, end, line_types=[s, c]))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle+rad(180), angle+rad(270), line_types=[s, c]))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width/2, left_turn_radius - lane_width/2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle+rad(0), angle+rad(-90), clockwise=False, line_types=[s, s]))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2)%4), StraightLane(start, end, line_types=[s, c]))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4), StraightLane(end, start, line_types=[n, c]))

        road = Road(network=net, np_random=self.np_random)
        self.road = road

    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        # Ego-vehicle
        ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
        ego_vehicle = MDPVehicle(self.road,
                                 ego_lane.position(0, 0),
                                 velocity=9,
                                 heading=ego_lane.heading_at(0)).plan_route_to("o1")
        MDPVehicle.SPEED_MIN = 0
        MDPVehicle.SPEED_MAX = 9
        MDPVehicle.SPEED_COUNT = 3
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle


def rad(deg):
    return deg*np.pi/180


register(
    id='intersection-v0',
    entry_point='highway_env.envs:IntersectionEnv',
)
