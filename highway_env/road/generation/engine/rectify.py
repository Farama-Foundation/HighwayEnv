from collections import defaultdict

import numpy as np

from ..spatial_hash import get_proximal_lanes_wrt_lane, lanes_spatial_hash
from .gen_utils import (
    Endpoint,
    Lane,
    do_line_segments_intersect,
    line_intersection_t,
    tqdm,
)


def rectify_map(
    lanes: list[Lane],
    merge_radius: int,
    forward_speed: int,
    disable_prints: bool = False,
) -> None:
    """
    Ensures proximal endpoints have the same string identifier
    and that intersecting lane paths are properly merged.
    Also removes any defective lanes.

    :param lanes: list of lanes
    :param merge_radius: distance at which an endpoint will join with another
    endpoint
    :param forward_speed: agent speed from the swarm generation process
    :param disable_prints: disables progress and status printing
    """
    rectify_short_lanes(lanes)
    conjoined_nodes = combine_nodes(
        lanes, merge_radius, mark=True, disable_prints=disable_prints
    )
    split_lanes(
        lanes,
        conjoined_nodes,
        merge_radius=merge_radius,
        forward_speed=forward_speed,
        disable_prints=disable_prints,
    )
    rectify_short_lanes(lanes)  # again
    combine_nodes(lanes, merge_radius, disable_prints=disable_prints)
    remove_identical_reference_lanes(lanes)
    prune_intersecting_lanes(lanes, disable_prints=disable_prints)


def rectify_short_lanes(lanes: list[Lane]) -> None:
    """
    Ensures all lanes are at least 3 points long

    :param lanes: list of lanes
    """
    lanes_to_remove = []
    for lane in lanes:
        if len(lane.points) <= 1:
            lanes_to_remove.append(lane)
        elif len(lane.points) == 2:
            a = lane.points[0]
            b = lane.points[1]
            lane.points.insert(1, (a + b) / 2)

    for dying_lane in lanes_to_remove:
        lanes[:] = [lane for lane in lanes if lane is not dying_lane]


def combine_nodes(
    lanes: list[Lane],
    merge_radius: int = 20,
    mark: bool = False,
    disable_prints: bool = False,
) -> None | list[str]:
    """
    Causes neighboring nodes to coalesce into the same logical
    intersection with the same identifier

    :param lanes: list of lanes
    :param merge_radius: distance at which an endpoint will join with
    another endpoint
    :param mark: if set to True, does not alter any existing nodes,
    but instead keeps track of which nodes would be combined
    :param disable_prints: disables progress and status printing

    :return: returns list of nodes that are proximal to other nodes
    [if mark is True; otherwise returns None]
    """

    lane_to_grid, grid_to_lanes = lanes_spatial_hash(
        lanes, gridsize=max(merge_radius, 50), use_boundaries=False
    )

    if mark:
        conjoined_nodes = []
    else:
        node_power = defaultdict(int)

    for laneID, lane in enumerate(
        tqdm(lanes, disabled=disable_prints, desc="Merging nodes")
    ):
        proximal_lanes = get_proximal_lanes_wrt_lane(
            laneID, lane_to_grid, grid_to_lanes, extended=True
        )
        for other_id in proximal_lanes:
            other_lane = lanes[other_id]
            for loc in ["start", "end"]:
                for other_loc in ["start", "end"]:
                    p0 = lane.points[Endpoint.l_to_i[loc]]
                    p1 = other_lane.points[Endpoint.l_to_i[other_loc]]
                    dist = np.linalg.norm(p0 - p1)
                    if dist < merge_radius:
                        lane_loc = getattr(lane, loc)
                        other_lane_loc = getattr(other_lane, other_loc)
                        if mark:
                            # We need to first ensure that no lane
                            # runs in between these two nodes
                            obstruction_found = False
                            for foreign_id in proximal_lanes:
                                foreign_lane = lanes[foreign_id]
                                pos_pairs = zip(
                                    foreign_lane.points,
                                    foreign_lane.points[1:],
                                )
                                for fp0, fp1 in pos_pairs:
                                    t_a, t_b = line_intersection_t(
                                        p0, p1 - p0, fp0, fp1 - fp0
                                    )
                                    if (
                                        t_a > 0.01
                                        and t_a < 0.99
                                        and t_b > 0.01
                                        and t_b < 0.99
                                    ):
                                        obstruction_found = True
                                        break
                            if not obstruction_found:
                                conjoined_nodes.append(lane_loc)
                        else:
                            if node_power[other_lane_loc] > node_power[lane_loc]:
                                setattr(lane, loc, other_lane_loc)
                                node_power[other_lane_loc] += 1
                            else:
                                setattr(other_lane, other_loc, lane_loc)
                                node_power[lane_loc] += 1

    if mark:
        return conjoined_nodes


def split_lanes(
    lanes: list[Lane],
    conjoined_nodes: list[str],
    merge_radius: int,
    forward_speed: int,
    disable_prints: bool = False,
) -> None:
    """
    Creates new intersections for lanes that ram into the middle
    of other lanes.

    :param lanes: list of lanes
    :param conjoined_nodes: list of nodes that are proximal to other nodes
    :param merge_radius: distance at which an endpoint will join with another
    lane
    :param forward_speed: agent speed from the swarm generation process
    :param disable_prints: disables progress and status printing
    """
    cutoff_length = np.ceil(merge_radius * 2.0 / forward_speed)

    child_parent_relationships = {}

    for lane in tqdm(
        lanes,
        disabled=disable_prints,
        desc="Creating intersections between proximal lanes",
    ):
        if len(lane.points) == 0:
            continue
        for loc in ["start", "end"]:
            if getattr(lane, loc) in conjoined_nodes:
                continue
            loc_pos = lane.points[Endpoint.l_to_i[loc]]

            for other_lane in lanes:
                if (
                    child_parent_relationships.get(lane) is other_lane
                    or child_parent_relationships.get(other_lane) is lane
                ):
                    continue
                found_index = -1
                closest_dist = None
                for i, pos in enumerate(other_lane.points):
                    dist = np.linalg.norm(pos - loc_pos)
                    if (
                        lane is not other_lane
                        or (i > cutoff_length and i < len(lane.points) - cutoff_length)
                    ) and (closest_dist is None or dist < closest_dist):
                        found_index = i
                        closest_dist = dist

                if closest_dist is not None and closest_dist < merge_radius:
                    if found_index < 2:
                        found_index = 2
                    if found_index > len(other_lane.points) - 2:
                        found_index = len(other_lane.points) - 2

                    lane_loc = getattr(lane, loc)
                    # 'old' means the earlier part of agent history
                    # / the bottom half of points
                    older_half = other_lane.points[:found_index]
                    other_lane.points = other_lane.points[found_index:]
                    old_start = other_lane.start
                    other_lane.start = lane_loc

                    new_lane = Lane(start=old_start, end=lane_loc, points=older_half)
                    lanes.append(new_lane)
                    child_parent_relationships[new_lane] = other_lane

                    conjoined_nodes.append(lane_loc)
                    break


def remove_identical_reference_lanes(lanes: list[Lane]) -> None:
    """
    Removing lanes whose start and end location is the same

    :param lanes: list of lanes
    """
    lanes_to_remove = []
    for lane in lanes:
        if lane.start == lane.end:
            lanes_to_remove.append(lane)

    for dying_lane in lanes_to_remove:
        lanes[:] = [lane for lane in lanes if lane is not dying_lane]


def prune_intersecting_lanes(lanes: list[Lane], disable_prints: bool = False) -> None:
    """
    Deleting lanes that cross over each other

    :param lanes: list of lanes
    :param disable_prints: disables progress and status printing
    """
    lane_to_grid, grid_to_lanes = lanes_spatial_hash(
        lanes, gridsize=50, use_boundaries=False
    )

    lanes_to_remove = []
    for laneID, lane in enumerate(
        tqdm(
            lanes,
            disabled=disable_prints,
            desc="Pruning Intersecting Lanes...",
        )
    ):
        proximal_lanes = get_proximal_lanes_wrt_lane(
            laneID, lane_to_grid, grid_to_lanes
        )
        collision_detected = False
        for other_id in proximal_lanes:
            if laneID < other_id:
                other_lane = lanes[other_id]
                pairs = zip(lane.points, lane.points[1:])
                for p0, p1 in pairs:
                    other_pairs = zip(other_lane.points, other_lane.points[1:])
                    for op0, op1 in other_pairs:
                        if do_line_segments_intersect(p0, p1, op0, op1):
                            collision_detected = True
                            break
                if collision_detected:
                    break
        if collision_detected:
            lanes_to_remove.append(lane)

    for dying_lane in lanes_to_remove:
        lanes[:] = [lane for lane in lanes if lane is not dying_lane]
