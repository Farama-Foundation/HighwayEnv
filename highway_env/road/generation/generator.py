import numpy as np

from .engine import (
    Lane,
    check_lanes_type_validity,
    correct_junction_boundaries,
    generate_lane_boundaries,
    generate_road_network_skeleton,
    get_invalid_lanes,
    get_nodeset,
    kill_lanes,
    rectify_map,
    remove_disjoint_clusters,
    seal_dead_end,
    twist_optimize,
)


def default_params() -> dict:
    """
    :return: Dict of parameters for procedural road generation:
    - **target_num_endpoints** - Number of endpoints to generate
    - **forward_speed** - length of individual lane line segments
    - **age_of_maturity** - timesteps before an agent can replicate or die
    - **lane_width** - Lane width, uniform across all lanes
    - **perlin_variation_params** - Perlin noise bounds for the following
    spatially varying attributes:
    **jitteriness** (erraticity of how agents turn left/right),
    **max_turn_speed** (approximate angular velocity of an agent),
    **replication_chance** (the tendency of forks in the road to occur), and
    **spontaneous_death_chance** (the tendency of dead-ends to occur)
    - **disable_prints** - Determines whether to include printed progress
    indicators
    - **seed** - Integer seed for the internal random number generator.
    ``None`` for a random seed
    """
    return {
        "target_num_endpoints": 2,
        "forward_speed": 10,
        "age_of_maturity": 4,
        "lane_width": 10,
        "perlin_variation_params": {
            "jitteriness": {"upper": 0.1, "lower": 0.0},
            "max_turn_speed": {"upper": 4.0, "lower": 0.01},
            "replication_chance": {"upper": 0.7, "lower": 0.0},
            "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
        },
        "disable_prints": True,
    }


def generate_random_lanes(
    rng: np.random.Generator, provided_params: dict | None = None
) -> list[Lane]:
    """
    Generates a procedurally generated lane network.

    :param rng: Random number generator
    :param provided_params: Generation parameters dict (optional)
    :return: list of lanes
    """
    params = default_params()
    if provided_params is not None:
        params.update(provided_params)

    merge_radius = params["forward_speed"] * 2
    prevent_replication_radius = params["forward_speed"] * params["age_of_maturity"]

    twist_iterations = params["forward_speed"] * 2
    twist_step = 0.0002 / params["forward_speed"]

    # Phase 1: Random swarm generation
    lanes = generate_road_network_skeleton(
        target_num_endpoints=max(2, params["target_num_endpoints"]),
        forward_speed=params["forward_speed"],
        merge_radius=merge_radius,
        prevent_replication_radius=prevent_replication_radius,
        age_of_maturity=params["age_of_maturity"],
        perlin_variation_params=params["perlin_variation_params"],
        rng=rng,
        disable_prints=params["disable_prints"],
    )

    # Phase 2: Rectification
    rectify_map(
        lanes,
        merge_radius=merge_radius + params["forward_speed"],
        forward_speed=params["forward_speed"],
        disable_prints=params["disable_prints"],
    )

    # Phase 3: Optimization
    twist_optimize(
        lanes,
        iterations=twist_iterations,
        step=twist_step,
        lane_width=params["lane_width"],
        disable_prints=params["disable_prints"],
    )

    # Phase 4: Boundary creation
    generate_lane_boundaries(lanes, params["lane_width"])
    for node in sorted(get_nodeset(lanes)):
        correct_junction_boundaries(lanes, node)
        seal_dead_end(lanes, node)

    # Phase 5: Validation
    invalids = get_invalid_lanes(
        lanes, params["forward_speed"], rng=rng, disable_prints=params["disable_prints"]
    )
    if not params["disable_prints"]:
        print(f"Removing {len(invalids)} obstructed lanes")
    kill_lanes(lanes, invalids)
    remove_disjoint_clusters(lanes)

    assert check_lanes_type_validity(lanes)

    return lanes


def serialize_lanes(lanes: list[Lane]) -> list[dict]:
    """
    Converts a Lane to a json-ready list of dicts.
    """
    lanes_serialized = []

    for lane in lanes:
        lane_serialized = {
            "start": lane.start,
            "end": lane.end,
            "points": [],
            "left_points": [],
            "right_points": [],
        }
        for pt in lane.points:
            lane_serialized["points"].append((pt[0], pt[1]))
        for pt in lane.left_points:
            lane_serialized["left_points"].append((pt[0], pt[1]))
        for pt in lane.right_points:
            lane_serialized["right_points"].append((pt[0], pt[1]))

        lanes_serialized.append(lane_serialized)

    return lanes_serialized


def unserialize_lanes(lanes_serialized: list[dict]) -> list[Lane]:
    """
    Converts a list of Lane-dicts to a list of Lane.
    """
    lanes = []

    for lane_serialized in lanes_serialized:
        new_lane = Lane(start=lane_serialized["start"], end=lane_serialized["end"])

        for pt in lane_serialized["points"]:
            new_lane.points.append(np.array([pt[0], pt[1]]))

        if "left_points" in lane_serialized:
            for pt in lane_serialized["left_points"]:
                new_lane.left_points.append(np.array([pt[0], pt[1]]))
            for pt in lane_serialized["right_points"]:
                new_lane.right_points.append(np.array([pt[0], pt[1]]))

        lanes.append(new_lane)

    return lanes


def save_lanes_to_disk(filename: str, lanes: list[Lane]):
    """
    Saves a list of lanes directly to a binary .npz file.
    """
    data = {}

    for i, lane in enumerate(lanes):
        data[f"lane_{i}_nodes"] = np.array([lane.start, lane.end])
        data[f"lane_{i}_points"] = np.asarray(lane.points)
        data[f"lane_{i}_left"] = np.asarray(lane.left_points)
        data[f"lane_{i}_right"] = np.asarray(lane.right_points)

    np.savez_compressed(filename, **data)


def load_lanes_from_disk(filename: str) -> list[Lane]:
    """
    Loads npz file and reconstructs the list of Lane objects.
    """
    with np.load(filename) as data:
        assert len(data.keys()) % 4 == 0
        num_lanes = int(len(data.keys()) / 4)
        lanes = []

        for i in range(num_lanes):
            start, end = data[f"lane_{i}_nodes"]
            new_lane = Lane(start=start, end=end)
            new_lane.points = data[f"lane_{i}_points"]
            new_lane.left_points = data[f"lane_{i}_left"]
            new_lane.right_points = data[f"lane_{i}_right"]

            lanes.append(new_lane)

    return lanes
