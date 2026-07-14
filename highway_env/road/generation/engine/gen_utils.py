from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from tqdm import tqdm as _tqdm


# Lane Graph Helpers ----------------------------------------


@dataclass
class Lane:
    """
    The raw geometry + logical connections of a lane.

    :param start: string identifier of the junction/node this lane originates
    from
    :param end: string identifier of the junction/node this lane terminates at
    :param points: ordered list of points of the centerline, from start to end
    :param left_points: ordered list of points of the left boundary
    :param right_points: ordered list of points of the right boundary
    """

    start: str
    end: str
    points: list[np.ndarray] = field(default_factory=list)
    left_points: list[np.ndarray] = field(default_factory=list)
    right_points: list[np.ndarray] = field(default_factory=list)

    def __str__(self):
        lines = [f"start: {self.start} end: {self.end}", "left:"]
        for pt in self.left_points:
            lines.append(f"{pt[0]} {pt[1]}")
        lines.append("right:")
        for pt in self.right_points:
            lines.append(f"{pt[0]} {pt[1]}")

        return "\n".join(lines)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass
class Endpoint:
    """
    Represents one end of a specific lane.

    :param id: index of Lane in lanes list
    :param loc: short for 'location': which end of the lane this represents
    (either 'start' or 'end')
    """

    id: int
    loc: str

    # converts a loc value to either the first or last index of a list
    l_to_i: ClassVar[dict[str, int]] = {"start": 0, "end": -1}

    def point_index(self) -> int:
        """
        :return: Either -1 or 0 depending on our loc field
        """
        return Endpoint.l_to_i[self.loc]

    def second_point_index(self) -> int:
        """
        :return: Either -2 or 1 depending on our loc field
        """
        return self.point_index() * 3 + 1  # transforms {-1, 0} to {-2, 1}

    def position(self, lanes: list[Lane]) -> np.ndarray:
        """
        :param lanes: list of lanes
        :return: position of endpoint based on the corresponding end of
        the lane's centerline
        """
        return lanes[self.id].points[self.point_index()]

    def vector_raw(self, lanes: list[Lane]) -> np.ndarray:
        """
        :param lanes: list of lanes
        :return: vector from second-last to last point of centerline
        """
        pos = self.position(lanes)
        pos2 = lanes[self.id].points[self.second_point_index()]
        return pos - pos2

    def vector(self, lanes: list[Lane]) -> np.ndarray:
        """
        :param lanes: list of lanes
        :return: unit vector denoting the direction the endpoint faces in
        """
        vec = self.vector_raw(lanes)
        return vec / np.linalg.norm(vec)


def get_radially_sorted_endpoints(lanes: list[Lane], node: str) -> list[Endpoint]:
    """
    :param lanes: list of lanes
    :param node: string identifier of junction
    :return: list of all the endpoints that make up the junction,
    sorted radially wrt the center of the junction
    """
    endpoints = []

    for i, lane in enumerate(lanes):
        for loc in ["start", "end"]:
            if getattr(lane, loc) == node:
                endpoints.append(Endpoint(id=i, loc=loc))

    if len(endpoints) == 0:
        return []

    midpoint = get_junction_pos(lanes, endpoints)

    def getTheta(ep):
        pos = ep.position(lanes) - midpoint
        return np.arctan2(pos[1], pos[0])

    endpoints.sort(key=getTheta)

    return endpoints


def get_junction_pos(
    lanes: list[Lane],
    junction: list[Endpoint],
    excluded_endpoint: Endpoint | None = None,
    use_boundaries: bool = False,
) -> np.ndarray:
    """
    :param lanes: list of lanes
    :param junction: list of all endpoints of the same intersection
    :param excluded_endpoint: endpoint to exclude when calculating the midpoint
    [optional]
    :param use_boundaries: use physical boundary points instead of centerline
    points
    :return: midpoint of all end boundary points
    """
    if excluded_endpoint is None:
        assert len(junction) > 0
    else:
        assert len(junction) > 1
        assert excluded_endpoint in junction

    pt = np.zeros(2)
    for ep in junction:
        if ep != excluded_endpoint:
            if use_boundaries:
                pt += lanes[ep.id].left_points[ep.point_index()]
                pt += lanes[ep.id].right_points[ep.point_index()]
            else:
                pt += ep.position(lanes)

    pt /= len(junction) - (1 if excluded_endpoint is not None else 0)
    if use_boundaries:
        pt /= 2

    return pt


def get_nodeset(lanes: list[Lane]):
    """
    :param lanes: list of lanes
    :return: set of all unique string junction identifiers
    """
    nodeset = set()
    for lane in lanes:
        nodeset.add(lane.start)
        nodeset.add(lane.end)

    return nodeset


# Geometry Helpers ----------------------------------------


def line_intersection_t(
    a: np.ndarray, av: np.ndarray, b: np.ndarray, bv: np.ndarray
) -> tuple[float, float]:
    """
    :param a: point on line A
    :param av: directional vector of line A
    :param b: point on line B
    :param bv: directional vector of line B
    :return: t_a and t_b, denoting the values of t that satisfy
    A + t_a*A_v = B + t_b*B_v
    """
    A = np.column_stack((av, -bv))
    B = b - a

    try:
        t_a, t_b = np.linalg.solve(A, B)
        return t_a, t_b
    except np.linalg.LinAlgError:
        return 0.0, 0.0


def do_line_segments_intersect(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> bool:
    """
    :param a0: first endpoint of line A
    :param a1: second endpoint of line A
    :param b0: first endpoint of line B
    :param b1: second endpoint of line B
    :return: whether or not the line segments intersect
    """
    t_a, t_b = line_intersection_t(a0, a1 - a0, b0, b1 - b0)
    return t_a >= 0 and t_a <= 1 and t_b >= 0 and t_b <= 1


def find_line_intersection(
    a: np.ndarray,
    av: np.ndarray,
    b: np.ndarray,
    bv: np.ndarray,
    return_t: bool = False,
) -> np.ndarray | tuple[np.ndarray, float, float]:
    """
    :param a: point on line A
    :param av: directional vector of line A
    :param b: point on line B
    :param bv: directional vector of line B
    :param return_t: whether or not to return the t values that satisfy
    A + t_a*A_v = B + t_b*B_v
    :return: the point of intersection. If return_t is True, then it is
    a tuple containing the intersection point and the two t values
    """
    t_a, t_b = line_intersection_t(a, av, b, bv)

    pt = a + (av * t_a)

    if return_t:
        return pt, t_a, t_b
    else:
        return pt


# Misc ----------------------------------------


def tqdm(iterable=None, disabled=False, *args, **kwargs):
    if not disabled:
        return _tqdm(iterable, *args, **kwargs)
    elif iterable is not None:
        return iterable
    else:
        return _DummyPbar()


class _DummyPbar:
    def __init__(self, *args, **kwargs):
        self.n = 0

    def update(self, n=1):
        self.n += n

    def refresh(self):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
