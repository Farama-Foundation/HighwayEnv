import numpy as np

from .gen_utils import Lane, find_line_intersection, get_radially_sorted_endpoints


def generate_lane_boundaries(lanes: list[Lane], lane_width: int) -> None:
    """
    Generates lateral left and right boundary points.

    :param lanes: list of lanes
    :param lane_width: intended lane width
    """
    for lane in lanes:
        lane.left_points = []
        lane.right_points = []

        for i, point in enumerate(lane.points):
            d = []
            if i != 0:
                d.append(lane.points[i - 1] - point)
            if i != len(lane.points) - 1:
                d.append(lane.points[i + 1] - point)

            v = d[0].copy()
            if i != 0:
                v *= -1

            lat = np.zeros(2)
            for _ in range(2):
                if len(d) == 1:
                    lat = np.array([-d[0][1], d[0][0]])
                    break
                elif len(d) == 2:
                    d[0] /= np.linalg.norm(d[0])
                    d[1] /= np.linalg.norm(d[1])

                    lat = (d[0] + d[1]) / 2.0

                    if np.linalg.norm(lat) == 0:
                        d.pop()
                        continue

                    break
                else:
                    assert False

            mag = np.linalg.norm(lat)
            lat *= (lane_width / 2) / mag

            if (lat[0] * v[1] - lat[1] * v[0]) < 0:
                lane.right_points.append(point + lat)
                lane.left_points.append(point - lat)
            else:
                lane.right_points.append(point - lat)
                lane.left_points.append(point + lat)


def correct_junction_boundaries(lanes: list[Lane], node: str) -> None:
    """
    Aligns the corner between angularly-adjacent lanes of a junction
    so that their boundary edges meet at one shared point.

    :param lanes: list of lanes
    :param node: string identifier of junction/intersection
    """
    junction = get_radially_sorted_endpoints(lanes, node)
    if len(junction) <= 1:
        return

    # Rule: Your left side should join with your left neighbor's right side
    # right-hand neighbor: up an index
    # left-hand neighbor: down an index
    # Left and right switches depending on which way the lane is oriented
    for epID, ep in enumerate(junction):
        other_ep = junction[epID - 1]

        self_side = "right_points"
        other_side = "left_points"
        if ep.loc == "start":
            self_side = "left_points"
        if other_ep.loc == "start":
            other_side = "right_points"

        self_side_list = getattr(lanes[ep.id], self_side)
        other_side_list = getattr(lanes[other_ep.id], other_side)

        # Trims boundary points until both points sit behind each
        # other relative to their own forward directions
        while True:
            pos = self_side_list[ep.point_index()]
            dir = ep.vector(lanes)

            other_pos = other_side_list[other_ep.point_index()]
            other_dir = other_ep.vector(lanes)

            vecToOther = other_pos - pos
            dot1 = vecToOther @ dir
            dot2 = -(vecToOther @ other_dir)
            if (
                dot1 > 0
                or dot2 > 0
                or len(self_side_list) <= 3
                or len(other_side_list) <= 3
            ):
                break

            self_side_list.pop(ep.point_index())
            other_side_list.pop(other_ep.point_index())

        new_pos = find_line_intersection(pos, dir, other_pos, other_dir)

        # If the new point does not sit in between the two original points,
        # we instead use a simple average of the two points
        b = other_pos - pos
        a = new_pos - pos

        if b @ b != 0:
            a1 = (a @ b) / (b @ b)

            if a1 <= 0 or a1 >= 1:
                new_pos = (pos + other_pos) / 2

        self_side_list[ep.point_index()] = new_pos
        other_side_list[other_ep.point_index()] = new_pos


def seal_dead_end(lanes: list[Lane], node: str) -> None:
    """
    Adds an additional right boundary point to seal off a dead-end junction.

    :param lanes: list of lanes
    :param node: string identifier of junction/intersetion
    """
    # A dead-end is a junction with only one endpoint
    junction = get_radially_sorted_endpoints(lanes, node)
    if len(junction) != 1:
        return

    ep = junction[0]
    lane = lanes[ep.id]

    if ep.loc == "start":
        lane.right_points.insert(0, lane.left_points[0])
    else:
        lane.right_points.append(lane.left_points[-1])

    # Shortens the center point value so that it does
    #  not touch our newly added border segment
    lane.points[ep.point_index()] = (
        lane.points[ep.point_index()] + lane.points[ep.second_point_index()]
    ) / 2
