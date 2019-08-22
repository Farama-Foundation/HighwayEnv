from __future__ import division, print_function, absolute_import
from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.control import MDPVehicle


class IntersectionEnv(AbstractEnv):

    COLLISION_REWARD = -0.3
    HIGH_VELOCITY_REWARD = 1
    ARRIVED_REWARD = 1

    DURATION = 18

    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics"
        },
        "policy_frequency": 1,  # [Hz]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "incoming_vehicle_destination": None,
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.5, 0.6],
        "scaling": 5.5*1.3
    }

    ACTIONS = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    def __init__(self):
        super(IntersectionEnv, self).__init__()
        self.steps = 0
        self.reset()

    def _reward(self, action):
        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.HIGH_VELOCITY_REWARD * (self.vehicle.velocity_index == self.vehicle.SPEED_COUNT - 1)
        reward = self.ARRIVED_REWARD if self.has_arrived else reward
        return utils.remap(reward, [self.COLLISION_REWARD, self.ARRIVED_REWARD], [0, 1])

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        return self.vehicle.crashed or self.steps >= self.DURATION or self.has_arrived

    def reset(self):
        self._make_road()
        self._make_vehicles()
        self.steps = 0
        return super(IntersectionEnv, self).reset()

    def step(self, action):
        results = super(IntersectionEnv, self).step(action)
        self.steps += 1
        self._clear_vehicles()
        self._spawn_vehicles()
        return results

    def _make_road(self):
        """
            Make an 4-way intersection.

            The horizontal road has the right of way. More precisely, the levels of priority are:
                - 3 for horizontal straight lanes and right-turns
                - 1 for vertical straight lanes and right-turns
                - 2 for horizontal left-turns
                - 0 for vertical left-turns
            The code for nodes in the road network is:
            (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)
        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width+5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = rad(90 * corner)
            is_horizontal = (corner + 0) % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle+rad(180), angle+rad(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width/2, left_turn_radius - lane_width/2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle+rad(0), angle+rad(-90), clockwise=False,
                                      line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random)
        self.road = road

    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        for _ in range(6):
            self._spawn_vehicles()
            [(self.road.act(), self.road.step(0.1)) for _ in range(15)]

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_vehicles_type.DISTANCE_WANTED = 2  # Low jam distance
        other_vehicles_type.COMFORT_ACC_MAX = 6
        other_vehicles_type.COMFORT_ACC_MIN = -3

        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("o1", "ir1", 0),
                                                   longitudinal=0,
                                                   velocity=None)
        vehicle.plan_route_to("o3")
        self.road.vehicles.append(vehicle)

        # Ego-vehicle
        ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
        ego_vehicle = MDPVehicle(self.road,
                                 ego_lane.position(0, 0),
                                 velocity=ego_lane.speed_limit,
                                 heading=ego_lane.heading_at(0)).plan_route_to("o"+str(self.np_random.randint(4)))
        MDPVehicle.SPEED_MIN = 0
        MDPVehicle.SPEED_MAX = 9
        MDPVehicle.SPEED_COUNT = 3
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    def _spawn_vehicles(self):
        if self.np_random.rand() < 1-0.6:
            return
        position_deviation = 1
        velocity_deviation = 1
        route = self.np_random.choice(range(4), size=2, replace=False)
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                                   longitudinal=5 + self.np_random.randn()*position_deviation,
                                                   velocity=8 + self.np_random.randn()*velocity_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

    def _clear_vehicles(self):
        self.road.vehicles = [
            vehicle for vehicle in self.road.vehicles
            if vehicle is self.vehicle or
            (not("il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                and vehicle.lane.local_coordinates(vehicle.position)[0] >=
                vehicle.lane.length - 4*vehicle.LENGTH) and vehicle.route is not None)
        ]

    @property
    def has_arrived(self):
        return "il" in self.vehicle.lane_index[0] \
                      and "o" in self.vehicle.lane_index[1] \
                      and self.vehicle.lane.local_coordinates(self.vehicle.position)[0] >= \
                      self.vehicle.lane.length - 3*self.vehicle.LENGTH


def rad(deg):
    return deg*np.pi/180


register(
    id='intersection-v0',
    entry_point='highway_env.envs:IntersectionEnv',
)
