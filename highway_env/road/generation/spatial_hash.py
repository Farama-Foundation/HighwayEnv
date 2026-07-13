from collections import defaultdict
from itertools import chain

import numpy as np


def point_to_gridpoint(point: np.ndarray, gridsize: int) -> tuple[int, int]:
    """
    Converts a world coordinate to the corresponding grid coordinate
    for spatial hashing.

    :param point: world position
    :param gridsize: length of a grid box
    :return: grid tuple coordinate
    """
    return tuple(np.floor(point / gridsize).astype(int))


def lanes_spatial_hash(
    lanes: list, gridsize: int = 100, use_boundaries: bool = True
) -> tuple[defaultdict[set], defaultdict[set]]:
    """
    Partitions lanes into separate grids for significantly
    faster proximal checks

    :param lanes: list of lanes
    :param gridsize: length of a grid box
    :param use_boundaries: refer to boundary instead of centerline points
    (defaults to True)
    :return: **lane_to_grid** (maps lane indices to the gridpoints they occupy)
    and **grid_to_lane** (maps gridpoints to the indices of lanes that inhabit
    them)
    """

    lane_to_grid = defaultdict(set)
    grid_to_lanes = defaultdict(set)

    for laneID, lane in enumerate(lanes):
        if use_boundaries:
            pts = chain(lane.left_points, lane.right_points)
        else:
            pts = lane.points

        last_gridpoint = None
        for point in pts:
            gridpoint = point_to_gridpoint(point, gridsize)
            lane_to_grid[laneID].add(gridpoint)
            grid_to_lanes[gridpoint].add(laneID)

            # In the case that we traverse precisely diagonally,
            # skipping over a grid:
            if (
                last_gridpoint is not None
                and np.abs(gridpoint[0] - last_gridpoint[0]) == 1
                and np.abs(gridpoint[1] - last_gridpoint[1]) == 1
            ):
                gp1 = (gridpoint[0], last_gridpoint[1])
                gp2 = (last_gridpoint[0], gridpoint[1])
                lane_to_grid[laneID].update((gp1, gp2))
                grid_to_lanes[gp1].add(laneID)
                grid_to_lanes[gp2].add(laneID)
            last_gridpoint = gridpoint
    return lane_to_grid, grid_to_lanes


gridhash_offsets = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


def get_proximal_lanes_wrt_gridpoint(
    grid_to_lanes: defaultdict[set], gridpoint: int, extended: bool = False
) -> set:
    """
    :param grid_to_lanes: map from gridpoints to the lanes that inhabit them
    :param gridpoint: grid tuple coordinate
    :param extended: whether or not to count lanes in neighboring grids
    :return: set of proximal lane indices
    """
    proximal_lanes = set()
    for offset in gridhash_offsets if extended else [(0, 0)]:
        new_point = (gridpoint[0] + offset[0], gridpoint[1] + offset[1])
        proximal_lanes.update(grid_to_lanes[new_point])

    return proximal_lanes


def get_proximal_lanes_wrt_lane(
    laneID: int,
    lane_to_grid: defaultdict[set],
    grid_to_lanes: defaultdict[set],
    extended: bool = False,
) -> set:
    """
    :param laneID: index of reference lane
    :param lane_to_grid: map from lanes to the gridpoints they occupy
    :param grid_to_lanes: map from gridpoints to the lanes that inhabit them
    :param extended: whether or not to count lanes in neighboring grids
    :return: set of proximal lane indices
    """
    proximal_lanes = set()
    for gridpoint in lane_to_grid[laneID]:
        proximal_lanes.update(
            get_proximal_lanes_wrt_gridpoint(
                grid_to_lanes, gridpoint, extended=extended
            )
        )

    proximal_lanes.discard(laneID)

    return proximal_lanes
