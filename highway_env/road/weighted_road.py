from __future__ import annotations

import numpy as np
import logging
from typing_extensions import override

from highway_env.road.lanes.abstract_lanes import AbstractLane
from road import RoadNetwork, LaneIndex, Route

logger = logging.getLogger(__name__)

class WeightedRoadNetwork(RoadNetwork):
    graph: dict[str, dict[str, list[tuple[int, AbstractLane]]]]

    def __init__(self):
        super().__init__()

    @override
    def add_lane(self, _from: str, _to: str, weight: int, lane: AbstractLane) -> None:
        """
        A weighted edge in the road network. See `RoadNetwork.add_lane`.
        :param _from: outgoing edge from this node
        :param _to: incoming edge to this node
        :param weight: the weight of the lane
        :param lane: lane geometry
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(tuple((weight, lane)))

    @override
    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        Gets the lane corresponding to the given lane index.
        :param index: tuple (origin node, destination node, lane id on road)
        :return: lane geometry
        """
        _from, _to, _id = index
        if _id is None:
            pass
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id][1]

    @override
    def get_closest_lane_index(
        self, position: np.ndarray, heading: float | None = None
    ) -> LaneIndex:
        """
        Gets the lane index, closest to the given position.
        :param position: world position [m]
        :param heading: heading [rad]
        :return: lane index of closest lane
        """
        idxs, dists = [], []
        for _from, edges in self.graph.items():
            for _to, lanes in edges.items():
                for _id, (weight, lane) in enumerate(lanes):
                    dists.append(lane.distance_with_heading(position, heading))
                    idxs.append((_from, _to, _id))
        return idxs[int(np.argmin(dists))]

    @override
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
        :return: the index of the next lane to be followed when current lane is finished.
        """
        _from, _to, _id = current_index
        next_to = next_id = None
        # Pick next road according to planned route
        if route:
            if (
                    route[0][:2] == current_index[:2]
            ):  # We just finished the first step of the route, drop it.
                route.pop(0)
            if (
                    route and route[0][0] == _to
            ):  # Next road in route is starting at the end of current road.
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
                ]  # (next_to, next_id, distance)
                next_to, next_id, _ = min(lanes_dists, key=lambda x: x[-1])
            except KeyError:
                return current_index
        else:
            # If it is known, follow it and get the closest lane
            next_id, _ = self.next_lane_given_next_road(
                _from, _to, next_id, next_to, next_id, projected_position
            )
        return _to, next_to, next_id
