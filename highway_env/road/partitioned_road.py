from collections import defaultdict
from itertools import chain

import numpy as np

from highway_env.road.generation.spatial_hash import (
    get_proximal_lanes_wrt_gridpoint,
    point_to_gridpoint,
)
from highway_env.road.lane import AbstractLane, PolyLane
from highway_env.road.road import LaneIndex, RoadNetwork


class PartitionedRoadNetwork(RoadNetwork):
    grid_to_lanes: defaultdict[set]
    partition_gridsize: int

    def __init__(self, partition_gridsize=100):
        super().__init__()
        self.grid_to_lanes = defaultdict(set)
        self.partition_gridsize = partition_gridsize

    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        super().add_lane(_from, _to, lane)
        lane_index = (_from, _to, len(self.graph[_from][_to]) - 1)

        if isinstance(lane, PolyLane):
            left_pts = lane.left_boundary_points
            right_pts = lane.right_boundary_points
        else:
            left_pts = []
            right_pts = []
            for long in range(0, int(lane.length + 1)):
                width = lane.width_at(long)
                left_pts.append(
                    lane.position(long, -width)
                )  # intentionally 2 times as wide
                right_pts.append(lane.position(long, width))

        last_gridpoint = None
        for pt in chain(left_pts, right_pts):
            gridpoint = point_to_gridpoint(pt, self.partition_gridsize)
            self.grid_to_lanes[gridpoint].add(lane_index)
            # In the case that we traverse precisely diagonally, skipping
            # over a grid:
            if (
                last_gridpoint is not None
                and np.abs(gridpoint[0] - last_gridpoint[0]) == 1
                and np.abs(gridpoint[1] - last_gridpoint[1]) == 1
            ):
                self.grid_to_lanes[(gridpoint[0], last_gridpoint[1])].add(lane_index)
                self.grid_to_lanes[(last_gridpoint[0], gridpoint[1])].add(lane_index)

    def add_lane_bidirectional(self, _from: str, _to: str, lane: AbstractLane) -> None:
        self.add_lane(_from, _to, lane)
        super().add_lane(_to, _from, lane)
        self.reversed_lane_indices.append((_to, _from, len(self.graph[_to][_from]) - 1))

    def get_closest_lane_index(
        self, position: np.ndarray, heading: float | None = None
    ) -> LaneIndex:
        indices = list(
            get_proximal_lanes_wrt_gridpoint(
                self.grid_to_lanes,
                point_to_gridpoint(position, self.partition_gridsize),
                extended=True,
            )
        )

        distances = [
            self.get_lane(l_i).distance_with_heading(position, heading)
            for l_i in indices
        ]

        if len(indices) == 0:
            return super().get_closest_lane_index(position, heading)

        return indices[int(np.argmin(distances))]
