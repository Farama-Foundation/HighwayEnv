from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain

import numpy as np

from ..spatial_hash import (
    get_proximal_lanes_wrt_gridpoint,
    get_proximal_lanes_wrt_lane,
    lanes_spatial_hash,
    point_to_gridpoint,
)
from .gen_utils import (
    Endpoint,
    Lane,
    find_line_intersection,
    get_junction_pos,
    get_nodeset,
    get_radially_sorted_endpoints,
    line_intersection_t,
    tqdm,
)


def get_invalid_lanes(
    lanes: list[Lane],
    forward_speed: int,
    disable_prints: bool = False,
    rng: np.random.Generator = None,
) -> list[Lane]:
    """
    Determines which lanes are blocked/not traversible.

    :param lanes: list of lanes
    :param forward_speed: agent speed from the swarm generation process
    :param disable_prints: disables progress and status printing
    :param rng: random number generator
    :return: list of lanes that are invalid
    """
    if rng is None:
        rng = np.random.default_rng()

    gridsize = 20
    _, grid_to_lanes = lanes_spatial_hash(lanes, gridsize)

    invalids = []

    for lane in tqdm(
        lanes, disabled=disable_prints, desc="Checking lanes for blockages"
    ):
        start_junction = get_radially_sorted_endpoints(lanes, lane.start)
        if len(start_junction) == 1:
            start_junction_pos = start_junction[0].position(lanes)
        else:
            start_junction_pos = get_junction_pos(
                lanes, start_junction, use_boundaries=True
            )

        end_junction = get_radially_sorted_endpoints(lanes, lane.end)
        if len(end_junction) == 1:
            end_junction_pos = end_junction[0].position(lanes)
        else:
            end_junction_pos = get_junction_pos(
                lanes, end_junction, use_boundaries=True
            )

        valid = determine_lane_validity(
            lanes,
            lane,
            start_junction_pos,
            end_junction_pos,
            grid_to_lanes,
            gridsize,
            forward_speed,
            rng,
        )

        if not valid:
            invalids.append(lane)

    return invalids


def determine_lane_validity(
    lanes: list[Lane],
    lane: Lane,
    start_pt: np.ndarray,
    end_pt: np.ndarray,
    grid_to_lanes: defaultdict[set],
    gridsize: int,
    forward_speed: int,
    rng: np.random.Generator,
) -> bool:
    """
    Checks if a lane is traversible or not by simulating the motion of
    car-sized balls that are pulled through the tunnel and repelled by
    its walls.

    :param lanes: list of lanes
    :param lane: lane to check
    :param start_pt: spawn point of the balls
    :param end_pt: goal waypoint of the balls
    :param grid_to_lanes: maps spatially hashed gridpoints to lane indices
    for fast proximal checks
    :param gridsize: size of grid for spatial hashing
    :param forward_speed: agent speed from the swarm generation process;
    is used for estimating the length of a lane
    :param rng: random number generator; is used for altering the initial
    velocities of spawned balls
    """
    lane_length = len(lane.points) * forward_speed

    pathway = [start_pt] + lane.points + [end_pt]
    if np.array_equal(pathway[0], pathway[1]):
        pathway.pop(0)
    if np.array_equal(pathway[-1], pathway[-2]):
        pathway.pop()

    """
    FORCES:
    - Pulling force: leads ball along pathway to end_pt
    - Wall repelling force: pushes ball from proximal line barriers
    - Ball repelling force: pushes ball from other balls to encourage
        exploration
    - Inelastic line barrier collisions
    - Friction / drag
    """

    ball_radius = 2  # CONSTRAINT: 2*ball_radius > vehicle width
    pull_force = 0.3 / 4  # CONSTRAINT: pull_force/friction <= ball_radius
    friction = 0.2 / 4
    repel_force = (
        ball_radius * pull_force
    )  # CONSTRAINT: repel_force/ball_radius <= pull_force
    repel_radius = 5  # distance at which the wall-repelling force takes effect
    cross_particle_repel_force = repel_force

    average_speed = pull_force / friction

    # if a particle is in the same spot (within repel_radius distance away)
    # after this much time, it is considered 'dead'
    death_timestep_threshold = 20

    particles = []
    max_timesteps_cap = int(5 * lane_length / average_speed)
    timesteps_before_particle_spam = int(1.5 * lane_length / average_speed)
    max_population = 10
    timesteps_per_history_update = 5

    # We will start out with one particle. If enough time passes and we still
    # haven't reached the goal yet, we start recruiting a bunch more particles
    # for further exploration

    # Simulation is run until one of the following conditions is met:
    #  A: a particle reaches the goal
    #  B: all particles become trapped
    # or C: maximum allotted timesteps is reached

    reached_goal = False
    for timestep in range(max_timesteps_cap):
        # Particle spawning
        if len(particles) == 0 or (
            len(particles) < max_population
            and timestep > timesteps_before_particle_spam
        ):
            proposed_particle = BallParticle(
                start_pt.copy(),
                (np.zeros(2) if len(particles) == 0 else rng.uniform(-0.5, 0.5, 2)),
            )

            if not proposed_particle.intersects_with_any(particles, ball_radius):
                particles.append(proposed_particle)

        for par in particles:
            par.pull_force(pathway, pull_force)

            gridpoint = point_to_gridpoint(par.pos, gridsize)
            proximal_lanes = get_proximal_lanes_wrt_gridpoint(
                grid_to_lanes, gridpoint, extended=True
            )
            par.border_force(
                lanes, proximal_lanes, repel_radius, repel_force, ball_radius
            )

            par.neighbor_force(particles, cross_particle_repel_force, ball_radius)

            par.vel *= 1 - friction
            par.pos += par.vel

            if timestep % timesteps_per_history_update == 0:
                par.hist.append(par.pos.copy())

            if np.linalg.norm(par.pos - pathway[-1]) < ball_radius:
                reached_goal = True
                break

        if reached_goal:
            break

        # Check for particle death
        if len(particles) == max_population:
            all_dead = True
            indices_past = int(death_timestep_threshold / timesteps_per_history_update)
            for par in particles:
                if par.is_alive(indices_past, repel_radius):
                    all_dead = False
                    break
            if all_dead:
                break

    return reached_goal


@dataclass
class BallParticle:
    """
    Used by determine_lane_validity
    """

    pos: np.ndarray
    vel: np.ndarray
    hist: list[np.ndarray] = field(default_factory=list)

    def pull_force(self, pathway, pull_force):
        closest_i = 0
        closest_dist = np.linalg.norm(self.pos - pathway[closest_i])
        for i, pt in enumerate(pathway):
            if i == 0 or i == len(pathway) - 1:
                continue

            dist = np.linalg.norm(self.pos - pt)
            if dist < closest_dist:
                closest_i = i
                closest_dist = dist

        if closest_i == len(pathway) - 2:
            pull_vector = pathway[closest_i + 1] - self.pos
        else:
            pull_vector = pathway[closest_i + 1] - pathway[closest_i]

        pull_vector *= pull_force / np.linalg.norm(pull_vector)
        self.vel += pull_vector

    # Border repel force + collisions
    def border_force(
        self, lanes, proximal_lanes, repel_radius, repel_force, ball_radius
    ):
        repel_vector = np.zeros(2)

        for other_laneID in proximal_lanes:
            other_lane = lanes[other_laneID]

            left_pairs = zip(other_lane.left_points, other_lane.left_points[1:])
            right_pairs = zip(other_lane.right_points, other_lane.right_points[1:])
            for a, b in chain(left_pairs, right_pairs):
                ab = b - a
                ap = self.pos - a
                ab_sq_len = np.sum(ab**2)
                if ab_sq_len == 0:
                    to_ball = ap
                    distance = np.linalg.norm(ap)
                else:
                    t = np.dot(ap, ab) / ab_sq_len
                    t_clamped = np.clip(t, 0.0, 1.0)
                    closest_point = a + t_clamped * ab
                    to_ball = self.pos - closest_point
                    distance = np.linalg.norm(to_ball)

                if distance < repel_radius:
                    repel_vector += (
                        to_ball * repel_force / max(distance, ball_radius) ** 2
                    )

                if distance < ball_radius:
                    if distance == 0:
                        if ab_sq_len == 0:  # should really never happen
                            if np.sum(self.vel**2) > 0:
                                normal = -self.vel
                            else:
                                normal = np.ones(2)
                        else:
                            normal = np.array([-ab[1], ab[0]])

                        normal /= np.linalg.norm(normal)
                    else:
                        normal = to_ball / distance

                    # Adjust position
                    self.pos += normal * (ball_radius - distance)

                    # Cancel velocity
                    vel_normal_magnitude = np.dot(self.vel, normal)
                    if vel_normal_magnitude < 0:
                        self.vel -= vel_normal_magnitude * normal

        self.vel += repel_vector

    # Ball-ball repel force
    def neighbor_force(self, particles, cross_particle_repel_force, ball_radius):
        repel_vector = np.zeros(2)
        for other_par in particles:
            if self is not other_par:
                vec = self.pos - other_par.pos
                vec *= (
                    cross_particle_repel_force
                    / max(np.linalg.norm(vec), ball_radius) ** 2
                )
                repel_vector += vec
        self.vel += repel_vector

    def is_alive(self, indices_past, repel_radius):
        if len(self.hist) < indices_past:
            return True
        displacement = np.linalg.norm(self.pos - self.hist[-indices_past])
        if displacement >= repel_radius:
            return True
        return False

    def intersects_with_any(self, particles, ball_radius):
        for par in particles:
            if np.linalg.norm(par.pos - self.pos) < ball_radius:
                return True
        return False


def kill_lanes(lanes: list[Lane], lanes_to_kil: list[Lane]) -> None:
    """
    Removes selected lanes and surgically repairs the holes in the
    lane borders left behind

    :param lanes: list of lanes
    :param lanes_to_kil: lanes to be removed
    """
    lane_ids_to_kil = [
        i for i, lane in enumerate(lanes) if id(lane) in {id(l) for l in lanes_to_kil}
    ]
    lane_ids_to_kil.sort(reverse=True)

    affected_nodes = defaultdict(list)
    for laneID in lane_ids_to_kil:
        for loc in ["start", "end"]:
            node = getattr(lanes[laneID], loc)
            if len(get_radially_sorted_endpoints(lanes, node)) > 1:
                affected_nodes[node].append(
                    (
                        lanes[laneID].left_points[Endpoint.l_to_i[loc]],
                        lanes[laneID].right_points[Endpoint.l_to_i[loc]],
                    )
                )

    for laneID in lane_ids_to_kil:
        del lanes[laneID]

    for node, segments in affected_nodes.items():
        junction = get_radially_sorted_endpoints(lanes, node)
        if len(junction) != 0:
            for segment in segments:
                closest_ep = None
                closest_side = None
                closest_dist = None
                first_point_is_p0 = False  # True -> p1 is first point
                for ep in junction:
                    for side in ["left_points", "right_points"]:
                        point = getattr(lanes[ep.id], side)[ep.point_index()]
                        dist0 = np.linalg.norm(segment[0] - point)
                        dist1 = np.linalg.norm(segment[1] - point)
                        if (
                            closest_dist is None
                            or dist0 < closest_dist
                            or dist1 < closest_dist
                        ):
                            closest_ep = ep
                            closest_side = side
                            closest_dist = min(dist0, dist1)
                            first_point_is_p0 = dist0 < dist1

                if closest_ep.loc == "start":
                    getattr(lanes[closest_ep.id], closest_side).insert(
                        0, segment[1] if first_point_is_p0 else segment[0]
                    )
                else:
                    getattr(lanes[closest_ep.id], closest_side).append(
                        segment[1] if first_point_is_p0 else segment[0]
                    )

    return affected_nodes


def remove_disjoint_clusters(lanes: list[Lane]) -> None:
    """
    Ensures all intersections are logically interconnected with
    each other by simply discarding any smaller disjoint sections
    of the network
    :param lanes: list of lanes
    """
    nodeset = get_nodeset(lanes)

    partition = []
    while len(nodeset) > 0:
        traversed = traverse_lane_graph(lanes, next(iter(nodeset)))
        partition.append(traversed)
        nodeset -= traversed

    for partition_element in partition:
        if len(partition_element) > len(nodeset):
            nodeset = partition_element

    # nodeset now contains the largest partition-element.
    # We must remove all lanes who does not connect to these nodes
    lane_ids_to_remove = []
    for laneID, lane in enumerate(lanes):
        if lane.start not in nodeset:
            lane_ids_to_remove.append(laneID)

    for laneID in reversed(lane_ids_to_remove):
        lanes.pop(laneID)


def traverse_lane_graph(lanes: list[Lane], node: str) -> set[str]:
    """
    :param lanes: list of lanes
    :param node: starting node
    :return: set of all nodes accessible from the start node
    """
    nodeset = {node}
    laneset = set()

    prev_laneset_size = -1
    while len(laneset) != prev_laneset_size:
        prev_laneset_size = len(laneset)
        for laneID, lane in enumerate(lanes):
            if lane.start in nodeset or lane.end in nodeset:
                nodeset.add(lane.start)
                nodeset.add(lane.end)

                laneset.add(laneID)

    return nodeset


def get_all_intersection_points(
    lanes: list[Lane],
    lane_to_grid: defaultdict[set],
    grid_to_lanes: defaultdict[set],
    disable_prints: bool = False,
) -> list[np.ndarray]:
    """
    Finds and lists any unwanted intersections between physical boundary lines.

    :param lanes: list of lanes
    :param lane_to_grid: maps lane indices to spatially hashed gridpoints
    for fast proximal checks
    :param grid_to_lanes: maps spatially hashed gridpoints to lane indices
    :param disable_prints: disables progress and status printing
    :return: list of intersection points
    """
    intersecting_points = []
    for laneID, lane in enumerate(
        tqdm(lanes, disabled=disable_prints, desc="Flagging intersection points")
    ):
        proximal_lanes = get_proximal_lanes_wrt_lane(
            laneID, lane_to_grid, grid_to_lanes
        )
        for other_id in proximal_lanes:
            if laneID < other_id:
                other_lane = lanes[other_id]
                left_pairs = zip(lane.left_points, lane.left_points[1:])
                right_pairs = zip(lane.right_points, lane.right_points[1:])
                for p0, p1 in chain(left_pairs, right_pairs):
                    other_left_pairs = zip(
                        other_lane.left_points, other_lane.left_points[1:]
                    )
                    other_right_pairs = zip(
                        other_lane.right_points, other_lane.right_points[1:]
                    )
                    for op0, op1 in chain(other_left_pairs, other_right_pairs):
                        t_a, t_b = line_intersection_t(p0, p1 - p0, op0, op1 - op0)
                        if t_a > 0.01 and t_a < 0.99 and t_b > 0.01 and t_b < 0.99:
                            intersecting_points.append(
                                find_line_intersection(p0, p1 - p0, op0, op1 - op0)
                            )

    return intersecting_points


def check_lanes_type_validity(lanes: list[Lane]) -> bool:
    """
    Checks type validity of every Lane.

    :param lanes: list of lanes
    :return: whether or not all lanes are valid
    """
    for lane in lanes:
        if not (
            isinstance(lane, Lane)
            and isinstance(lane.start, str)
            and isinstance(lane.end, str)
            and isinstance(lane.points, list)
            and isinstance(lane.left_points, list)
            and isinstance(lane.right_points, list)
        ):
            return False

        for pt in chain(lane.points, lane.left_points, lane.right_points):
            if not (isinstance(pt, np.ndarray) and pt.shape == (2,)):
                return False

    return True
