from __future__ import division, print_function
import numpy as np
import pandas as pd

from highway_env.logger import Loggable
from highway_env.road.lane import LineType, StraightLane
from highway_env.vehicle.control import ControlledVehicle
from highway_env.vehicle.dynamics import Obstacle


class RoadNetwork(object):
    def __init__(self):
        self.graph = {}

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def add_lane(self, _from, _to, lane):
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index):
        _from, _to, _id = index
        return self.graph[_from][_to][_id]

    def get_lane_index(self, position):
        """
            Get the index of the lane closest to a world position.

        :param position: a world position [m]
        :return: the index of the closest lane
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    s, r = l.local_coordinates(position)
                    distances.append(abs(r) + max(s-l.length, 0) + max(0-s, 0))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane(self, current_index, plan=[]):
        _from, _to, _id = current_index
        if plan:
            raise NotImplementedError()
        else:
            try:
                next_to = list(self.graph[_to].keys())[np.random.randint(len(self.graph[_to]))]
                if len(self.graph[_from][_to]) == len(self.graph[_from][_to]):
                    next_id = _id
                else:
                    next_id = np.random.randint(len(self.graph[_to][next_to]))
            except KeyError as e:
                print(e)
                return current_index
        return _to, next_to, next_id

    def road_lanes(self, lane_index):
        return self.graph[lane_index[0]][lane_index[1]]

    def neighbour_lanes(self, lane_index):
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def is_same_road(lane_index_1, lane_index_2):
        return lane_index_1[0] == lane_index_2[0] and lane_index_1[1] == lane_index_2[1]


class Road(Loggable):
    """
        A road is a set of lanes, and a set of vehicles driving on these lanes
    """

    def __init__(self, lanes=None, network=None, vehicles=None):
        """
            New road.

        :param lanes: the lanes composing the road
        :param vehicles: the vehicles driving on the road
        """
        self.lanes = lanes or []
        self.network = network or []
        self.vehicles = vehicles or []

    @classmethod
    def create_random_road(cls,
                           lanes_count,
                           lane_width=4,
                           vehicles_count=50,
                           vehicles_type=ControlledVehicle,
                           np_random=None):
        """
            Create a road composed of straight adjacent lanes with randomly located vehicles on it.

        :param lanes_count: number of lanes
        :param lane_width: lanes width [m]
        :param vehicles_count: number of vehicles on the road
        :param vehicles_type: type of vehicles on the road
        :param np.random.RandomState np_random: a random number generator
        :return: the created road
        """
        lanes = []
        for lane in range(lanes_count):
            origin = np.array([0, lane * lane_width])
            heading = 0
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes_count - 1 else LineType.NONE]
            lanes.append(StraightLane(origin, heading, lane_width, line_types))
        r = Road(lanes)
        r.add_random_vehicles(vehicles_count, vehicles_type, np_random)
        return r

    def add_random_vehicles(self, vehicles_count=50, vehicles_type=ControlledVehicle, np_random=None):
        """
            Create some new random vehicles of a given type, and add them on the road.

        :param vehicles_count: number of vehicles to create
        :param vehicles_type: type of vehicles to create
        :param np.random.RandomState np_random: a random number generator
        """
        for _ in range(vehicles_count):
            self.vehicles.append(vehicles_type.create_random(self, np_random=np_random))

    def close_vehicles_to(self, vehicle, distances):
        return [v for v in self.vehicles if (distances[0] < vehicle.lane_distance_to(v) < distances[1]
                                             and v is not vehicle)]

    def closest_vehicles_to(self, vehicle, count):
        sorted_v = sorted([v for v in self.vehicles
                           if v is not vehicle
                           and -2*vehicle.LENGTH < vehicle.lane_distance_to(v)],
                          key=lambda v: abs(vehicle.lane_distance_to(v)))
        return sorted_v[:count]

    def act(self):
        """
            Decide the actions of each entity on the road.
        """
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt):
        """
            Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        for vehicle in self.vehicles:
            vehicle.step(dt)
            for other in self.vehicles:
                vehicle.check_collision(other)

    def neighbour_vehicles(self, vehicle, lane=None):
        """
            Find the preceding and following vehicles of a given vehicle.
        :param vehicle: the vehicle whose neighbours must be found
        :param lane: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane = lane or vehicle.lane
        if not lane:
            return None, None
        s = lane.local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles:
            if v is not vehicle and v.lane == lane:
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    def dump(self):
        """
            Dump the data of all entities on the road
        """
        for v in self.vehicles:
            if not isinstance(v, Obstacle):
                v.dump()

    def get_log(self):
        """
            Concatenate the logs of all entities on the road.
        :return: the concatenated log.
        """
        return pd.concat([v.get_log() for v in self.vehicles])

    def __repr__(self):
        return self.vehicles.__repr__()
