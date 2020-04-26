import numpy as np
import pandas as pd
import logging

from highway_env.logger import Loggable
from highway_env.road.lane import LineType, StraightLane
from highway_env.vehicle.kinematics import Obstacle

logger = logging.getLogger(__name__)


class RoadNetwork(object):
    def __init__(self):
        self.graph = {}

    def add_node(self, node):
        """
            A node represents an symbolic intersection in the road network.
        :param node: the node label.
        """
        if node not in self.graph:
            self.graph[node] = []

    def add_lane(self, _from, _to, lane):
        """
            A lane is encoded as an edge in the road network.
        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index):
        """
            Get the lane geometry corresponding to a given index in the road network.
        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry.
        """
        _from, _to, _id = index
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id]

    def get_closest_lane_index(self, position):
        """
            Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance(position))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane(self, current_index, route=None, position=None, np_random=np.random):
        """
            Get the index of the next lane that should be followed after finishing the current lane.

            If a plan is available and matches with current lane, follow it.
            Else, pick next road randomly.
            If it has the same number of lanes as current road, stay in the same lane.
            Else, pick next road's closest lane.
        :param current_index: the index of the current lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
        _from, _to, _id = current_index
        next_to = None
        # Pick next road according to planned route
        if route:
            if route[0][:2] == current_index[:2]:  # We just finished the first step of the route, drop it.
                route.pop(0)
            if route and route[0][0] == _to:  # Next road in route is starting at the end of current road.
                _, next_to, route_id = route[0]
            elif route:
                logger.warning("Route {} does not start after current road {}.".format(route[0], current_index))
        # Randomly pick next road
        if not next_to:
            try:
                next_to = list(self.graph[_to].keys())[np_random.randint(len(self.graph[_to]))]
            except KeyError:
                # logger.warning("End of lane reached.")
                return current_index

        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(lanes,
                          key=lambda l: self.get_lane((_to, next_to, l)).distance(position))

        return _to, next_to, next_id

    def bfs_paths(self, start, goal):
        """
            Breadth-first search of all routes from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: list of paths from start to goal.
        """
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in set(self.graph[node].keys()) - set(path):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def shortest_path(self, start, goal):
        """
            Breadth-first search of shortest path from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest path from start to goal.
        """
        try:
            return next(self.bfs_paths(start, goal))
        except StopIteration:
            return None

    def all_side_lanes(self, lane_index):
        """
        :param lane_index: the index of a lane.
        :return: all indexes of lanes belonging to the same road.
        """
        return self.graph[lane_index[0]][lane_index[1]]

    def side_lanes(self, lane_index):
        """
                :param lane_index: the index of a lane.
                :return: indexes of lanes next to a an input lane, to its right or left.
                """
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def is_same_road(lane_index_1, lane_index_2, same_lane=False):
        """
            Is lane 1 in the same road as lane 2?
        """
        return lane_index_1[:2] == lane_index_2[:2] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    @staticmethod
    def is_leading_to_road(lane_index_1, lane_index_2, same_lane=False):
        """
            Is lane 1 leading to of lane 2?
        """
        return lane_index_1[1] == lane_index_2[0] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    def is_connected_road(self, lane_index_1, lane_index_2, route=None, same_lane=False, depth=0):
        """
            Is the lane 2 leading to a road within lane 1's route?

            Vehicles on these lanes must be considered for collisions.
        :param lane_index_1: origin lane
        :param lane_index_2: target lane
        :param route: route from origin lane, if any
        :param same_lane: compare lane id
        :param depth: search depth from lane 1 along its route
        :return: whether the roads are connected
        """
        if RoadNetwork.is_same_road(lane_index_2, lane_index_1, same_lane) \
                or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # Route is starting at current road, skip it
                return self.is_connected_road(lane_index_1, lane_index_2, route[1:], same_lane, depth)
            elif route and route[0][0] == lane_index_1[1]:
                # Route is continuing from current road, follow it
                return self.is_connected_road(route[0], lane_index_2, route[1:], same_lane, depth - 1)
            else:
                # Recursively search all roads at intersection
                _from, _to, _id = lane_index_1
                return any([self.is_connected_road((_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1)
                            for l1_to in self.graph.get(_to, {}).keys()])
        return False

    def lanes_list(self):
        return [lane for tos in self.graph.values() for ids in tos.values() for lane in ids]

    @staticmethod
    def straight_road_network(lanes=4, length=10000, angle=0):
        net = RoadNetwork()
        for lane in range(lanes):
            origin = np.array([0, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE]
            net.add_lane(0, 1, StraightLane(origin, end, line_types=line_types))
        return net

    def position_heading_along_route(self, route, longitudinal, lateral):
        """
            Get the absolute position and heading along a route composed of several lanes at some local coordinates.
        :param route: a planned route, list of lane indexes
        :param longitudinal: longitudinal position
        :param lateral: : lateral position
        :return: position, heading
        """
        while len(route) > 1 and longitudinal > self.get_lane(route[0]).length:
            longitudinal -= self.get_lane(route[0]).length
            route = route[1:]
        return self.get_lane(route[0]).position(longitudinal, lateral), self.get_lane(route[0]).heading_at(longitudinal)


class Road(Loggable):
    """
        A road is a set of lanes, and a set of vehicles driving on these lanes
    """

    def __init__(self, network=None, vehicles=None, obstacles=None, np_random=None, record_history=False):
        """
            New road.

        :param network: the road network describing the lanes
        :param vehicles: the vehicles driving on the road
        :param np.random.RandomState np_random: a random number generator for vehicle behaviour
        :param record_history: whether the recent trajectories of vehicles should be recorded for display
        """
        self.network = network or []
        self.vehicles = vehicles or []
        self.obstacles = obstacles or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history

    def close_vehicles_to(self, vehicle, distance, count=None, sort=False, see_behind=True):
        vehicles = [v for v in self.vehicles
                    if np.linalg.norm(v.position - vehicle.position) < distance
                    and v is not vehicle
                    and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))]
        if sort:
            vehicles = sorted(vehicles, key=lambda v: abs(vehicle.lane_distance_to(v)))
        if count:
            vehicles = vehicles[:count]
        return vehicles

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
        # TODO: create a shallow copy of vehicles list(vehicle.copy()) and pop crashed vehicles from it to reduce
        #  complexity and prevent multiple checks
        for vehicle in self.vehicles:
            for other in self.vehicles:
                vehicle.check_collision(other)
            for other in self.obstacles:
                vehicle.check_collision(other)

    def neighbour_vehicles(self, vehicle, lane_index=None):
        """
            Find the preceding and following vehicles of a given vehicle.
        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles:
            if v is not vehicle and True:  # self.network.is_connected_road(v.lane_index, lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
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
