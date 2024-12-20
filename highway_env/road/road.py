from __future__ import annotations

import logging
from collections import deque
from queue import Queue
from typing import TYPE_CHECKING, Deque, List, Tuple, overload
from typing_extensions import override

import networkx as nx
import numpy as np

from highway_env.road.lanes.abstract_lanes import AbstractLane
from highway_env.road.lanes.lane_utils import LaneType, LineType
from highway_env.road.lanes.unweighted_lanes import StraightLane, lane_from_config
from highway_env.vehicle.objects import Landmark


if TYPE_CHECKING:
    from highway_env.vehicle import kinematics, objects

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]


class PathException(Exception):
    pass


class RoadNetwork:
    graph: dict[str, dict[str, list[AbstractLane]]]

    def __init__(self):
        self.graph = {}

    def add_lane(
        self,
        _from: str,
        _to: str,
        lane: AbstractLane,
        weight: int = None,
        lane_type: LaneType = None,
    ) -> None:
        """
        A lane is encoded as an edge in the road network.

        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        :param weight: weight of the lane
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        Get the lane geometry corresponding to a given index in the road network.

        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry.
        """
        _from, _to, _id = index
        if _id is None:
            pass
        if _id is None or len(self.graph[_from][_to]) == 1:
            _id = 0

        return self.graph[_from][_to][_id]

    def get_closest_lane_index(
        self, position: np.ndarray, heading: float | None = None
    ) -> LaneIndex:
        """
        Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :param heading: a heading angle [rad].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane(
        self,
        current_index: LaneIndex,
        route: Route = None,
        position: np.ndarray = None,
        np_random: np.random.RandomState = np.random,
    ) -> LaneIndex:
        """
        Get the index of the next lane that should be followed after finishing the current lane.

        - If a plan is available and matches with current lane, follow it.
        - Else, pick next road randomly.
        - If it has the same number of lanes as current road, stay in the same lane.
        - Else, pick next road's closest lane.
        :param current_index: the index of the current target lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished
        """
        _from, _to, _id = current_index
        next_to = next_id = None
        # Pick next road according to planned route
        if route:
            if route[0][:2] == current_index[:2]:
                route.pop(0)
            if route and route[0][0] == _to:
                _, next_to, next_id = route[0]
            elif route:
                logger.warning(
                    "Route {} does not start after current road {}.".format(
                        route[0], current_index
                    )
                )

        # Compute current projected (desired) position
        long, lat = self.get_lane(current_index).local_coordinates(position)
        projected_position = self.get_lane(current_index).position(long, lateral=0)
        # If next route is not known
        if not next_to:
            # Pick the one with the closest lane to projected target position
            try:
                lanes_dists = [
                    (
                        next_to,
                        *self.next_lane_given_next_road(
                            _from, _to, _id, next_to, next_id, projected_position
                        ),
                    )
                    for next_to in self.graph[_to].keys()
                ]
                next_to, next_id, _ = min(lanes_dists, key=lambda x: x[-1])
            except KeyError:
                return current_index
        else:
            # If it is known, follow it and get the closest lane
            next_id, _ = self.next_lane_given_next_road(
                _from, _to, _id, next_to, next_id, projected_position
            )
        return _to, next_to, next_id

    def next_lane_given_next_road(
        self,
        _from: str,
        _to: str,
        _id: int,
        next_to: str,
        next_id: int,
        position: np.ndarray,
    ) -> tuple[int, float]:
        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            if next_id is None:
                next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(
                lanes, key=lambda l: self.get_lane((_to, next_to, l)).distance(position)
            )
        return next_id, self.get_lane((_to, next_to, next_id)).distance(position)

    def bfs_paths(self, start: str, goal: str) -> list[list[str]]:
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
            for _next in sorted(
                [key for key in self.graph[node].keys() if key not in path]
            ):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def shortest_path(self, start: str, goal: str) -> list[str]:
        """
        Breadth-first search of shortest path from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest path from start to goal.
        """
        return next(self.bfs_paths(start, goal), [])

    def all_side_lanes(self, lane_index: LaneIndex) -> list[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: all lanes belonging to the same road.
        """
        return [
            (lane_index[0], lane_index[1], i)
            for i in range(len(self.graph[lane_index[0]][lane_index[1]]))
        ]

    def side_lanes(self, lane_index: LaneIndex) -> list[LaneIndex]:
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
    def is_same_road(
        lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False
    ) -> bool:
        """Is lane 1 in the same road as lane 2?"""
        return lane_index_1[:2] == lane_index_2[:2] and (
            not same_lane or lane_index_1[2] == lane_index_2[2]
        )

    @staticmethod
    def is_leading_to_road(
        lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False
    ) -> bool:
        """Is lane 1 leading to of lane 2?"""
        return lane_index_1[1] == lane_index_2[0] and (
            not same_lane or lane_index_1[2] == lane_index_2[2]
        )

    def is_connected_road(
        self,
        lane_index_1: LaneIndex,
        lane_index_2: LaneIndex,
        route: Route = None,
        same_lane: bool = False,
        depth: int = 0,
    ) -> bool:
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
        if RoadNetwork.is_same_road(
            lane_index_2, lane_index_1, same_lane
        ) or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                return self.is_connected_road(
                    lane_index_1, lane_index_2, route[1:], same_lane, depth
                )
            elif route and route[0][0] == lane_index_1[1]:
                return self.is_connected_road(
                    route[0], lane_index_2, route[1:], same_lane, depth - 1
                )
            else:
                # Recursively search all roads at intersection
                _from, _to, _id = lane_index_1
                return any(
                    [
                        self.is_connected_road(
                            (_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1
                        )
                        for l1_to in self.graph.get(_to, {}).keys()
                    ]
                )
        return False

    def lanes_list(self) -> list[AbstractLane]:
        return [
            lane for to in self.graph.values() for ids in to.values() for lane in ids
        ]

    def lanes_dict(self) -> dict[str, AbstractLane]:
        return {
            (from_, to_, i): lane
            for from_, tos in self.graph.items()
            for to_, ids in tos.items()
            for i, lane in enumerate(ids)
        }

    @staticmethod
    def straight_road_network(
        lanes: int = 4,
        start: float = 0,
        length: float = 10_000,
        angle: float = 0,
        speed_limit: float = 30,
        nodes_str: tuple[str, str] | None = None,
        net: WeightedRoadnetwork | None = None,
        weight: int = 1,
    ) -> WeightedRoadnetwork:
        net = net or WeightedRoadnetwork()
        nodes_str = nodes_str or ("0", "1")
        for lane in range(lanes):
            origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start + length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array(
                [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
            )
            origin = rotation @ origin
            end = rotation @ end
            line_types = [
                LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE,
            ]
            net.add_lane(
                *nodes_str,
                lane=StraightLane(
                    origin, end, line_types=line_types, speed_limit=speed_limit
                ),
                weight=weight,
                lane_type=LaneType.HIGHWAY,
            )
        return net

    def position_heading_along_route(
        self,
        route: Route,
        longitudinal: float,
        lateral: float,
        current_lane_index: LaneIndex,
    ) -> tuple[np.ndarray, float]:
        """
        Get the absolute position and heading along a route composed of several lanes at some local coordinates.

        :param route: a planned route, list of lane indexes
        :param longitudinal: longitudinal position
        :param lateral: : lateral position
        :param current_lane_index: current lane index of the vehicle
        :return: position, heading
        """

        def _get_route_head_with_id(route_):
            lane_index_ = route_[0]
            if lane_index_[2] is None:
                # We know which road segment will be followed by the vehicle, but not which lane.
                # Hypothesis: the vehicle will keep the same lane_id as the current one.
                id_ = (
                    current_lane_index[2]
                    if current_lane_index[2]
                    < len(self.graph[current_lane_index[0]][current_lane_index[1]])
                    else 0
                )
                lane_index_ = (lane_index_[0], lane_index_[1], id_)
            return lane_index_

        lane_index = _get_route_head_with_id(route)
        while len(route) > 1 and longitudinal > self.get_lane(lane_index).length:
            longitudinal -= self.get_lane(lane_index).length
            route = route[1:]
            lane_index = _get_route_head_with_id(route)

        return self.get_lane(lane_index).position(longitudinal, lateral), self.get_lane(
            lane_index
        ).heading_at(longitudinal)

    def random_lane_index(self, np_random: np.random.RandomState) -> LaneIndex:
        _from = np_random.choice(list(self.graph.keys()))
        _to = np_random.choice(list(self.graph[_from].keys()))
        _id = np_random.integers(len(self.graph[_from][_to]))
        return _from, _to, _id

    @classmethod
    def from_config(cls, config: dict) -> None:
        net = cls()
        for _from, to_dict in config.items():
            net.graph[_from] = {}
            for _to, lanes_dict in to_dict.items():
                net.graph[_from][_to] = []
                for lane_dict in lanes_dict:
                    net.graph[_from][_to].append(lane_from_config(lane_dict))
        return net

    def to_config(self) -> dict:
        graph_dict = {}
        for _from, to_dict in self.graph.items():
            graph_dict[_from] = {}
            for _to, lanes in to_dict.items():
                graph_dict[_from][_to] = []
                for lane in lanes:
                    graph_dict[_from][_to].append(lane.to_config())
        return graph_dict


class WeightedRoadnetwork(RoadNetwork):
    graph_net: nx.MultiDiGraph

    def __init__(self):
        super().__init__()
        self.graph_net = nx.MultiDiGraph()

    def weight(self, u: str, v: str) -> float:
        """
        Returns the weight of an edge. Does not check whether the edge exists.
        """
        return self.graph_net[u][v][0]["weight"]

    def get_lane_type(self, u: str, v: str) -> LaneType:
        """
        Returns the lane type of an edge, (u, v).
        """
        return self.graph_net[u][v][0]["lane_type"]

    def topological_sort(self, source: str) -> list[tuple[str, str]]:
        """
        Sorts the graph topologically. Please note that this is not a proper topological sort, as that cannot be
        done on cyclic graphs. Each node is explored from the source.
        """
        sorted_seq: list[tuple[str, str]] = []
        queue = deque([source])
        while len(queue) > 0:
            u = queue.popleft()
            for v in self.graph_net[u]:
                if (u, v) not in sorted_seq:
                    sorted_seq.append((u, v))
                    queue.append(v)
        return sorted_seq

    def dijkstra(self, source: str, goal: str) -> list[str]:
        """
        Performs the Dijkstra shortest-path algorithm. Please note that it cannot handle negative weights, and will
        raise an error if a negative weight is detected.
        """
        dists: dict[str, float] = {}
        predecessors: dict[str, str] = {}
        queue = deque()
        for v in self.graph_net.nodes:
            queue.append(v)
            dists[v] = np.inf

        dists[source] = 0

        while len(queue) > 0:
            u = min(queue, key=dists.get)
            if u == goal:
                break
            queue.remove(u)
            neighbors = self.graph_net.neighbors(u)
            for v in neighbors:
                if self.weight(u, v) < 0:
                    raise ValueError(f"weight for edge ({u}, {v}) is less than zero")
                if dists[u] + self.weight(u, v) < dists[v]:
                    dists[v] = dists[u] + self.weight(u, v)
                    predecessors[v] = u

        path = deque([goal])
        while path[0] is not source:
            try:
                node = path[0]
                path.appendleft(predecessors[node])
            except KeyError:
                raise PathException(
                    f"Could not find path {source} ~> {goal}, attempted to access non-existing predecessor for {node} in {predecessors}"
                )

        return list(path)

    def bellman_ford_negative_cycle(
        self, source: str, goal: str, max_pi: int = 3
    ) -> list[str]:
        """
        Performs the Bellman-Ford shortest path algorithm, while disregarding negative weight cycles. The number of
        repeated visits to a single node can be set with the max_pi parameter.
        :param source: source node.
        :param goal: goal node.
        :param max_pi: the maximum number of predecessors for each node. 0 is unbounded length.
        """
        dists: dict[str, float] = {}
        predecessors: dict[str, deque[str]] = {}

        for vertex in self.graph_net.nodes:
            dists[vertex] = np.inf
            predecessors[vertex] = deque(maxlen=max_pi)

        dists[source] = 0

        # Exploring the graph
        for i in range(0, self.graph_net.order() - 1):
            pies = dict()
            for u, v in self.topological_sort(source):
                if dists[v] > dists[u] + self.weight(u, v):
                    dists[v] = dists[u] + self.weight(u, v)
                    if pies.get(v) is None:
                        pies[v] = u
            for vertex in pies.keys():
                # Only adding, if below the maximum length.
                if len(predecessors[vertex]) < max_pi:
                    predecessors[vertex].appendleft(pies.get(vertex))

        # Determining the path
        path = deque([goal])
        while path[0] is not source:
            try:
                node = path[0]
                path.appendleft(predecessors[node].pop())
            except IndexError:
                print(f"src: {source}, goal: {goal}\n predecessors: {predecessors}")
                raise PathException(
                    f"could not find path {source} ~> {goal}, attempted to pop from empty list. node: {node}, predecessors: {predecessors}"
                )

        return list(path)

    def bellman_ford(self, source: str, goal: str) -> list[str]:
        """
        Performs the Bellman-Ford shortest-path algorithm. This implementation will raise an error if a negative
        weight cycle is detected.
        """
        dists: dict[str, float] = {}
        predecessors: dict[str, str] = {}
        for v in self.graph_net.nodes:
            dists[v] = np.inf

        dists[source] = 0

        # Exploring the graph
        for i in range(0, self.graph_net.order() - 1):
            for u, v in self.topological_sort(source):
                if dists[v] > dists[u] + self.weight(u, v):
                    dists[v] = dists[u] + self.weight(u, v)
                    predecessors[v] = u

        # Detecting negative weight cycles
        for u, v in self.graph_net.edges():
            if dists[v] > dists[u] + self.weight(u, v):
                # Finding a vertex on the negative weight cycle
                predecessors[v] = u
                visited = deque([v])
                while u not in visited:
                    visited.append(u)
                    u = predecessors[u]

                # Finding the cycle that u is a part of
                negative_weight_cycle = deque([u])
                v = predecessors[u]
                while v != u:
                    negative_weight_cycle.appendleft(v)
                    v = predecessors[v]
                raise Exception(
                    f"Graph contains negative weight cycle: {list(negative_weight_cycle)}"
                )

        # Constructing the path
        path = deque([goal])
        while path[0] is not source:
            try:
                node = path[0]
                path.appendleft(predecessors[node])
            except KeyError:
                raise PathException(
                    f"Could not find path {source} ~> {goal}, attempted to access non-existing predecessor for {node} in {predecessors}"
                )

        return list(path)

    def shortest_path(self, start: str, goal: str) -> list[str]:
        return self.bellman_ford_negative_cycle(start, goal)

    def add_lane(
        self,
        _from: str,
        _to: str,
        lane: AbstractLane,
        weight: int = None,
        lane_type: LaneType = None,
    ) -> None:
        super().add_lane(_from, _to, lane)
        if weight is None:
            raise ValueError("Cannot create edge with weight None")
        if lane_type is None:
            raise ValueError("Cannot create edge with lane type None")

        if not self.graph_net.has_node(_from):
            self.graph_net.add_node(_from)
        if not self.graph_net.has_node(_to):
            self.graph_net.add_node(_to)

        # Adding the weight
        try:
            node = self.graph_net[_from][_to]
        except KeyError:  # Edge does not exists
            self.graph_net.add_edge(_from, _to, weight=weight, lane_type=lane_type)
            return


class Road:
    """A road is a set of lanes, and a set of vehicles driving on these lanes."""

    def __init__(
        self,
        network: RoadNetwork | WeightedRoadnetwork = None,
        vehicles: list[kinematics.Vehicle] = None,
        road_objects: list[objects.RoadObject] = None,
        np_random: np.random.RandomState = None,
        record_history: bool = False,
    ) -> None:
        """
        New road.

        :param network: the road network describing the lanes
        :param vehicles: the vehicles driving on the road
        :param road_objects: the objects on the road including obstacles and landmarks
        :param np.random.RandomState np_random: a random number generator for vehicle behaviour
        :param record_history: whether the recent trajectories of vehicles should be recorded for display
        """
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history

    def close_objects_to(
        self,
        vehicle: kinematics.Vehicle,
        distance: float,
        count: int | None = None,
        see_behind: bool = True,
        sort: bool = True,
        vehicles_only: bool = False,
    ) -> object:
        vehicles = [
            v
            for v in self.vehicles
            if np.linalg.norm(v.position - vehicle.position) < distance
            and v is not vehicle
            and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))
        ]
        obstacles = [
            o
            for o in self.objects
            if np.linalg.norm(o.position - vehicle.position) < distance
            and -2 * vehicle.LENGTH < vehicle.lane_distance_to(o)
        ]

        objects_ = vehicles if vehicles_only else vehicles + obstacles

        if sort:
            objects_ = sorted(objects_, key=lambda o: abs(vehicle.lane_distance_to(o)))
        if count:
            objects_ = objects_[:count]
        return objects_

    def close_vehicles_to(
        self,
        vehicle: kinematics.Vehicle,
        distance: float,
        count: int | None = None,
        see_behind: bool = True,
        sort: bool = True,
    ) -> object:
        return self.close_objects_to(
            vehicle, distance, count, see_behind, sort, vehicles_only=True
        )

    def act(self) -> None:
        """Decide the actions of each entity on the road."""
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt: float) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        for vehicle in self.vehicles:
            vehicle.step(dt)
        for i, vehicle in enumerate(self.vehicles):
            for other in self.vehicles[i + 1 :]:
                vehicle.handle_collisions(other, dt)
            for other in self.objects:
                vehicle.handle_collisions(other, dt)

    def neighbour_vehicles(
        self, vehicle: kinematics.Vehicle, lane_index: LaneIndex = None
    ) -> tuple[kinematics.Vehicle | None, kinematics.Vehicle | None]:
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
        for v in self.vehicles + self.objects:
            if v is not vehicle and not isinstance(v, Landmark):
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

    def __repr__(self):
        return self.vehicles.__repr__()
