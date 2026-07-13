import numpy as np

from .gen_utils import (
    Endpoint,
    Lane,
    get_junction_pos,
    get_nodeset,
    get_radially_sorted_endpoints,
    tqdm,
)


def twist_optimize(
    lanes: list[Lane],
    iterations: int = 40,
    step: float = 0.00001,
    n: int = 3,
    lane_width: int = 10,
    disable_prints: bool = False,
) -> None:
    """
    Uses gradient-descent to twist, squish, and rotate the endpoints
    of lanes at intersections so that they properly point at each other.

    :param lanes: list of lanes
    :param iterations: number of gradient-descent steps
    :param step: gradient-descent step size. highly sensitive
    :param n: number of actuators/joints to twist
    :param lane_width: intended lane width
    :param disable_prints: disables progress and status printing
    """

    rotate_optimize(lanes, n)

    r = 3
    nodeset = get_nodeset(lanes)

    for node in tqdm(nodeset, disabled=disable_prints, desc="Twisting Endpoints..."):
        junction = get_radially_sorted_endpoints(lanes, node)

        for _ in range(iterations):
            for ep in junction:
                length = len(lanes[ep.id].points)
                if length > n:
                    step_twist_gradient(lanes, junction, ep, step=step, n=n, r=r)
                elif length > 2:
                    step_twist_gradient(lanes, junction, ep, step=step, n=2, r=r)

        squish_optimize(lanes, junction, r)

    prune_redundant_lanes(lanes, lane_width)


def step_twist_gradient(
    lanes: list[Lane], junction: list, ep, step=0.0002, n=3, r=3
) -> None:
    """
    Twists an endpoint to minimize the loss:
    (x(a) + r * cos(theta(a)) - x_t)^2 + (y(a) + r * sin(theta(a)) - y_t)^2
    [with respect to a, the angle of twist applied], where:
    - x_t and y_t is the center of the junction
    - x(a) and y(a) are the coordinates of the endpoint after applying twist
    with angle a and are expressed as recursive functions:
        - x(i, a) = x(i-1, a) + r_i * cos(theta_i + a)
            - x(0, a) = anchor point
            - x(n, a) = endpoint
            - r_i = length of segment; theta_i = angle offset of segment
            relative to orientation of previous segment
    - theta(a) is the absolute heading of the tip of the endpoint
    - r is the length of an imaginary rigid appendage attached to the tip
    of the endpoint.
        - We want to get the end of the appendage to be as close to (x_t, y_t)
        as possible
    At a loss of near zero, the endpoint would naturally 'point'
    towards the center of our junction. The twisting motion
    ensures the curve remains smooth.

    :param lanes: list of lanes
    :param junction: list of all the endpoints that make up the junction
    :param ep: specific endpoint of the junction
    :param step: gradient-descent step size
    :param n: number of actuators/joints to twist
    :param r: length of imaginary appendage attached to the endpoint
    """
    if len(junction) <= 1:
        return 0

    # Computing x_t and y_t, x(a), and y(a)
    x_a, y_a = ep.position(lanes)
    vx_a, vy_a = ep.vector(lanes)
    theta_a = np.atan2(vy_a, vx_a)

    x_t, y_t = get_junction_pos(lanes, junction, excluded_endpoint=ep)

    for endpoint in junction:
        if endpoint is not ep:
            vec = endpoint.vector(lanes)
            x_t += r * vec[0] / (len(junction) - 1)
            y_t += r * vec[1] / (len(junction) - 1)

    # Computing x'(a, n) and y'(a, n):
    #   x'(i) = -r_i * sin(theta_i) + x'(i-1)
    #   y'(i) = r_i * cos(theta_i) + y'(i-1)
    #   x'(0) = y'(0) = 0
    polar_sequence = get_polar_sequence(lanes, ep, n)

    x_a_derivative = 0
    y_a_derivative = 0
    for i in range(1, n + 1):
        c = polar_sequence[i]
        x_a_derivative += -c[1] * np.sin(c[0])
        y_a_derivative += c[1] * np.cos(c[0])

    # Change in the heading of the endpoint's tip would be
    # equal to number of actuators * angle twist applied
    theta_a_derivative = n

    # Computing the loss gradient
    # L' = (x(a) + rcos(theta(a)) - x_t)(x'(a) - (rsin(theta(a)) * theta'(a)))
    #    + (y(a) + rsin(theta(a)) - y_t)(y'(a) + (rcos(theta(a)) * theta'(a)))
    loss_gradient = (x_a + r * np.cos(theta_a) - x_t) * (
        x_a_derivative - (r * np.sin(theta_a) * theta_a_derivative)
    ) + (y_a + r * np.sin(theta_a) - y_t) * (
        y_a_derivative + (r * np.cos(theta_a) * theta_a_derivative)
    )

    twist_endpoint(lanes, ep, -loss_gradient * step, n)


def twist_endpoint(lanes: list[Lane], ep: Endpoint, angle: float, n: int = 3) -> None:
    """
    Twists an endpoint by treating each line segment as an actuator that
    rotates a constant amount.

    :param lanes: list of lanes
    :param ep: endpoint to twist
    :param angle: direction and magnitude of twist
    :param n: number of actuators
    """
    polar_coord_sequence = get_polar_sequence(lanes, ep, n)

    for i, (theta, r) in enumerate(polar_coord_sequence):
        if i == 0:
            continue
        polar_coord_sequence[i] = (theta + angle * i, r)

    for i in range(1, n + 1):
        index = i_to_index(lanes, ep, n, i)
        base_index = i_to_index(lanes, ep, n, i - 1)

        x_offset = np.cos(polar_coord_sequence[i][0]) * polar_coord_sequence[i][1]
        y_offset = np.sin(polar_coord_sequence[i][0]) * polar_coord_sequence[i][1]
        base_point = lanes[ep.id].points[base_index]

        lanes[ep.id].points[index] = base_point + np.array([x_offset, y_offset])


def get_polar_sequence(lanes: list[Lane], ep: Endpoint, n: int) -> list[tuple]:
    """
    Converts the first/last n segments of an endpoint to a sequence of
    relative polar coordinate offsets.

    :param lanes: list of lanes
    :param ep: endpoint
    :param n: number of actuators
    """
    polar_coord_sequence = [(-999, -999)]  # the 0th index of this is invalid
    for i in range(1, n + 1):
        pos0 = lanes[ep.id].points[i_to_index(lanes, ep, n, i)]
        pos1 = lanes[ep.id].points[
            i_to_index(lanes, ep, n, i - 1)
        ]  # pos1 is 'closer to the base' than pos0
        vec = pos0 - pos1

        r = np.linalg.norm(vec)
        theta = np.atan2(vec[1], vec[0])

        polar_coord_sequence.append((theta, r))  # turning a constant amount

    return polar_coord_sequence


def i_to_index(lanes: list[Lane], ep: Endpoint, n: int, i: int) -> int:
    """
    Maps i from the recursive functions x(i) or y(i) to point indices.
    * i = 0 is the 'trunk' or base point
        * for 'start' endpoints, index becomes n
        * for 'end' endpoints, index becomes len(points)-n-1
    * i = n is the 'tip' or final point
        * for 'start' endpoints, index becomes 0
        * for 'end' endpoints, index becomes len(points)-1

    :param lanes: list of lanes
    :param ep: endoint
    :param n: number of actuators
    :param i: index for recursive functions x(i) & y(i)
    """

    if ep.loc == "start":
        return n - i
    else:
        return i + len(lanes[ep.id].points) - n - 1


def rotate_optimize(lanes: list[Lane], n: int = 3) -> None:
    """
    Directly rotates a lane if it is too short to be twisted

    :param lanes: list of lanes
    :param n: number of actuators used in twist step (threshold for shortness)
    """
    for laneID, lane in enumerate(lanes):
        if len(lane.points) <= n:
            start_junction = get_radially_sorted_endpoints(lanes, lane.start)
            if len(start_junction) > 1:
                start = get_junction_pos(
                    lanes,
                    start_junction,
                    excluded_endpoint=Endpoint(id=laneID, loc="start"),
                )
            else:
                start = start_junction[0].position(lanes)

            end_junction = get_radially_sorted_endpoints(lanes, lane.end)
            if len(end_junction) > 1:
                end = get_junction_pos(
                    lanes,
                    end_junction,
                    excluded_endpoint=Endpoint(id=laneID, loc="end"),
                )
            else:
                end = end_junction[0].position(lanes)

            for i in range(len(lane.points)):
                num_pts = len(lane.points)
                lane.points[i] = (end - start) * ((i + 1) / (num_pts + 1)) + start


def squish_optimize(lanes: list[Lane], junction: list[Endpoint], r: int) -> None:
    """
    Squishes an endpoint by either removing points or squeezing them
    closer to ensure an endpoint does not stretch beyond the center
    of a junction.

    :param lanes: list of lanes
    :param junction: list of all the endpoints that make up the junction
    :param r: desired distance offset from center of junction
    """
    if len(junction) <= 1:
        return

    for ep in junction:
        mid = get_junction_pos(lanes, junction, excluded_endpoint=ep)

        b = ep.vector_raw(lanes)

        for _ in range(
            5
        ):  # Done multiple times in case we have to remove several lane points
            # before we get aligned properly
            pos = ep.position(lanes)
            second_pos = lanes[ep.id].points[ep.second_point_index()]

            c = mid - second_pos

            if c @ c != 0:
                offset = r * c / np.linalg.norm(c)
            else:
                offset = r * b / np.linalg.norm(b)

            a = mid - offset - pos
            a1 = a @ b / (b @ b)

            if a1 < -1:
                if len(lanes[ep.id].points) > 2:
                    lanes[ep.id].points.pop(ep.point_index())
                    continue
            elif a1 < 0:
                new = (lanes[ep.id].points[ep.second_point_index()] + mid) * 0.5
                lanes[ep.id].points[ep.point_index()] = new

            break


def prune_redundant_lanes(lanes: list[Lane], lane_width: int) -> None:
    """
    Removing lanes that have the same start + end and overlap with each other.

    :param lanes: list of lanes
    :param lane_width: intended lane width
    """

    duplicate_found = True
    while duplicate_found:
        duplicate_found = False
        for lane in lanes:
            for other_lane in lanes:
                if lane is not other_lane:
                    if (
                        lane.start == other_lane.start and lane.end == other_lane.end
                    ) or (
                        lane.end == other_lane.start and lane.start == other_lane.end
                    ):
                        number_of_points_too_close = 0
                        for point in lane.points:
                            closest_distance = None
                            for point2 in other_lane.points:
                                dist = np.linalg.norm(point - point2)
                                if closest_distance is None or dist < closest_distance:
                                    closest_distance = dist
                            if (
                                closest_distance is not None
                                and closest_distance < lane_width * 1.5
                            ):
                                number_of_points_too_close += 1

                        if number_of_points_too_close > 2:
                            duplicate_found = True
                            lanes[:] = [l for l in lanes if l is not other_lane]
                        break
            if duplicate_found:
                break
