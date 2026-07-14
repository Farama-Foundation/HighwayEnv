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
        "target_num_endpoints": 100,
        "forward_speed": 10,
        "age_of_maturity": 4,
        "lane_width": 10,
        "perlin_variation_params": {
            "jitteriness": {"upper": 0.1, "lower": 0.0},
            "max_turn_speed": {"upper": 4.0, "lower": 0.01},
            "replication_chance": {"upper": 0.7, "lower": 0.0},
            "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
        },
        "disable_prints": False,
        "seed": None,
    }


def generate_random_lanes(params: dict | None = None) -> list[Lane]:
    """
    Generates a procedurally generated lane network.

    :param params: Generation parameters dict (optional)
    :return: list of lanes
    """
    if params is None:
        params = default_params()

    rng = np.random.default_rng(params["seed"])

    merge_radius = params["forward_speed"] * 2
    prevent_replication_radius = params["age_of_maturity"] * params["forward_speed"]

    twist_iterations = 2 * params["forward_speed"]
    twist_step = 0.0002 / params["forward_speed"]

    disable_prints = params["disable_prints"]

    # Phase 1: Random swarm generation
    lanes = generate_road_network_skeleton(
        target_num_endpoints=max(2, params["target_num_endpoints"]),
        forward_speed=params["forward_speed"],
        merge_radius=merge_radius,
        prevent_replication_radius=prevent_replication_radius,
        age_of_maturity=params["age_of_maturity"],
        perlin_variation_params=params["perlin_variation_params"],
        disable_prints=disable_prints,
        rng=rng,
    )

    # Phase 2: Rectification
    rectify_map(
        lanes,
        merge_radius=merge_radius,
        forward_speed=params["forward_speed"],
        disable_prints=disable_prints,
    )

    # Phase 3: Optimization
    twist_optimize(
        lanes,
        iterations=twist_iterations,
        step=twist_step,
        lane_width=params["lane_width"],
        disable_prints=disable_prints,
    )

    # Phase 4: Boundary creation
    generate_lane_boundaries(lanes, params["lane_width"])
    for node in get_nodeset(lanes):
        correct_junction_boundaries(lanes, node)
        seal_dead_end(lanes, node)

    # Phase 5: Validation
    invalids = get_invalid_lanes(
        lanes, params["forward_speed"], disable_prints=disable_prints, rng=rng
    )
    if not disable_prints:
        print(f"Removing {len(invalids)} obstructed lanes")
    kill_lanes(lanes, invalids)
    remove_disjoint_clusters(lanes)

    assert check_lanes_type_validity(lanes)

    return lanes


def print_lanes(lanes: list[Lane]) -> None:
    """
    Prints a list of lanes.
    """
    for _, lane in enumerate(lanes):
        print(lane)


def serialize_lanes(lanes: list[Lane]) -> list[dict]:
    """
    Converts a Lane to a json-ready list of dicts.
    """
    lanes_s = []

    for lane in lanes:
        lane_s = {}
        lane_s["start"] = lane.start
        lane_s["end"] = lane.end

        lane_s["points"] = []
        lane_s["left_points"] = []
        lane_s["right_points"] = []

        for i, pt in enumerate(lane.points):
            lane_s["points"].append((pt[0], pt[1]))
        for i, pt in enumerate(lane.left_points):
            lane_s["left_points"].append((pt[0], pt[1]))
        for i, pt in enumerate(lane.right_points):
            lane_s["right_points"].append((pt[0], pt[1]))

        lanes_s.append(lane_s)

    return lanes_s


def unserialize_lanes(lanes_s: list[dict]) -> list[Lane]:
    """
    Converts a list of Lane-dicts to a list of Lane.
    """
    lanes = []

    for lane_s in lanes_s:
        new_lane = Lane(start=lane_s["start"], end=lane_s["end"])

        for pt in lane_s["points"]:
            new_lane.points.append(np.array([pt[0], pt[1]]))

        if "left_points" in lane_s:
            for pt in lane_s["left_points"]:
                new_lane.left_points.append(np.array([pt[0], pt[1]]))
            for pt in lane_s["right_points"]:
                new_lane.right_points.append(np.array([pt[0], pt[1]]))

        lanes.append(new_lane)

    return lanes


def save_lanes_to_disk(filename: str, lanes: list[Lane]):
    """
    Saves a list of lanes directly to a binary .npz file.
    """
    data_to_save = {}

    for i, lane in enumerate(lanes):
        data_to_save[f"lane_{i}_metadata"] = np.array([lane.start, lane.end])
        data_to_save[f"lane_{i}_points"] = np.asarray(lane.points)
        data_to_save[f"lane_{i}_left"] = np.asarray(lane.left_points)
        data_to_save[f"lane_{i}_right"] = np.asarray(lane.right_points)

    np.savez_compressed(filename, **data_to_save)


def load_lanes_from_disk(filename: str) -> list[Lane]:
    """
    Loads npz file and reconstructs the list of Lane objects.
    """
    with np.load(filename) as data:
        lanes = []
        num_lanes = sum(1 for key in data.keys() if key.endswith("_metadata"))

        for i in range(num_lanes):
            metadata = data[f"lane_{i}_metadata"]

            new_lane = Lane(start=metadata[0], end=metadata[1])
            new_lane.points = data[f"lane_{i}_points"]
            new_lane.left_points = data[f"lane_{i}_left"]
            new_lane.right_points = data[f"lane_{i}_right"]

            lanes.append(new_lane)

    return lanes
