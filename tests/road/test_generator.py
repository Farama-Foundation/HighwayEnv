import numpy as np
import pytest

from highway_env.road.generation.engine.gen_utils import Lane
from highway_env.road.generation.engine.validation import get_all_intersection_points
from highway_env.road.generation.generator import (
    generate_random_lanes,
    load_lanes_from_disk,
    save_lanes_to_disk,
    serialize_lanes,
    unserialize_lanes,
)
from highway_env.road.generation.spatial_hash import lanes_spatial_hash


GENERATOR_CASES = [
    pytest.param(None, False, id="default"),
    pytest.param(
        {
            "target_num_endpoints": 10,
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
        },
        True,
        id="small",
    ),
    pytest.param(
        {
            "target_num_endpoints": 1000,
            "forward_speed": 5,
            "age_of_maturity": 4,
            "lane_width": 10,
            "perlin_variation_params": {
                "jitteriness": {"upper": 0.2, "lower": 0.0},
                "max_turn_speed": {"upper": 5.0, "lower": 0.01},
                "replication_chance": {"upper": 0.7, "lower": 0.2},
                "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
            },
            "disable_prints": False,
        },
        False,
        id="big_chaotic",
    ),
    pytest.param(
        {
            "target_num_endpoints": 100,
            "forward_speed": 20,
            "age_of_maturity": 4,
            "lane_width": 10,
            "perlin_variation_params": {
                "jitteriness": {"upper": 0.1, "lower": 0.0},
                "max_turn_speed": {"upper": 4.0, "lower": 0.01},
                "replication_chance": {"upper": 0.7, "lower": 0.0},
                "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
            },
            "disable_prints": False,
        },
        False,
        id="fast",
    ),
    pytest.param(
        {
            "target_num_endpoints": 100,
            "forward_speed": 5,
            "age_of_maturity": 4,
            "lane_width": 10,
            "perlin_variation_params": {
                "jitteriness": {"upper": 0.1, "lower": 0.0},
                "max_turn_speed": {"upper": 4.0, "lower": 0.01},
                "replication_chance": {"upper": 0.7, "lower": 0.0},
                "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
            },
            "disable_prints": False,
        },
        False,
        id="slow",
    ),
    pytest.param(
        {
            "target_num_endpoints": 100,
            "forward_speed": 10,
            "age_of_maturity": 4,
            "lane_width": 20,
            "perlin_variation_params": {
                "jitteriness": {"upper": 0.1, "lower": 0.0},
                "max_turn_speed": {"upper": 4.0, "lower": 0.01},
                "replication_chance": {"upper": 0.7, "lower": 0.0},
                "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
            },
            "disable_prints": False,
        },
        False,
        id="wide",
    ),
    pytest.param(
        {
            "target_num_endpoints": 100,
            "forward_speed": 10,
            "age_of_maturity": 4,
            "lane_width": 5,
            "perlin_variation_params": {
                "jitteriness": {"upper": 0.1, "lower": 0.0},
                "max_turn_speed": {"upper": 4.0, "lower": 0.01},
                "replication_chance": {"upper": 0.7, "lower": 0.0},
                "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
            },
            "disable_prints": False,
        },
        False,
        id="skinny",
    ),
    pytest.param(
        {
            "target_num_endpoints": 100,
            "forward_speed": 10,
            "age_of_maturity": 4,
            "lane_width": 10,
            "perlin_variation_params": {
                "jitteriness": {"upper": 1.0, "lower": 0.5},
                "max_turn_speed": {"upper": 4.0, "lower": 0.01},
                "replication_chance": {"upper": 0.7, "lower": 0.0},
                "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
            },
            "disable_prints": False,
        },
        False,
        id="jittery",
    ),
    pytest.param(
        {
            "target_num_endpoints": 100,
            "forward_speed": 10,
            "age_of_maturity": 4,
            "lane_width": 10,
            "perlin_variation_params": {
                "jitteriness": {"upper": 0.01, "lower": 0.0},
                "max_turn_speed": {"upper": 5.0, "lower": 0.01},
                "replication_chance": {"upper": 0.7, "lower": 0.0},
                "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
            },
            "disable_prints": False,
        },
        False,
        id="smooth",
    ),
    pytest.param(
        {
            "target_num_endpoints": 100,
            "forward_speed": 10,
            "age_of_maturity": 4,
            "lane_width": 10,
            "perlin_variation_params": {
                "jitteriness": {"upper": 0.1, "lower": 0.0},
                "max_turn_speed": {"upper": 6.0, "lower": 2.0},
                "replication_chance": {"upper": 0.7, "lower": 0.0},
                "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
            },
            "disable_prints": False,
        },
        False,
        id="twisty",
    ),
    pytest.param(
        {
            "target_num_endpoints": 100,
            "forward_speed": 10,
            "age_of_maturity": 4,
            "lane_width": 10,
            "perlin_variation_params": {
                "jitteriness": {"upper": 0.1, "lower": 0.0},
                "max_turn_speed": {"upper": 0.1, "lower": 0.0},
                "replication_chance": {"upper": 0.7, "lower": 0.0},
                "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
            },
            "disable_prints": False,
        },
        False,
        id="straight",
    ),
    pytest.param(
        {
            "target_num_endpoints": 100,
            "forward_speed": 10,
            "age_of_maturity": 4,
            "lane_width": 10,
            "perlin_variation_params": {
                "jitteriness": {"upper": 0.1, "lower": 0.0},
                "max_turn_speed": {"upper": 6.0, "lower": 2.0},
                "replication_chance": {"upper": 1.0, "lower": 6.0},
                "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
            },
            "disable_prints": False,
        },
        False,
        id="hyperdense",
    ),
    pytest.param(
        {
            "target_num_endpoints": 100,
            "forward_speed": 10,
            "age_of_maturity": 4,
            "lane_width": 10,
            "perlin_variation_params": {
                "jitteriness": {"upper": 0.1, "lower": 0.0},
                "max_turn_speed": {"upper": 6.0, "lower": 2.0},
                "replication_chance": {"upper": 0.1, "lower": 0.1},
                "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
            },
            "disable_prints": False,
        },
        False,
        id="sparse",
    ),
    pytest.param(
        {
            "target_num_endpoints": 100,
            "forward_speed": 10,
            "age_of_maturity": 4,
            "lane_width": 10,
            "perlin_variation_params": {
                "jitteriness": {"upper": 0.1, "lower": 0.0},
                "max_turn_speed": {"upper": 6.0, "lower": 2.0},
                "replication_chance": {"upper": 0.7, "lower": 0.0},
                "spontaneous_death_chance": {"upper": 1.0, "lower": 1.0},
            },
            "disable_prints": False,
        },
        False,
        id="deadly",
    ),
]


def lanelist_equality(lanes1: list[Lane], lanes2: list[Lane]):
    for lane1, lane2 in zip(lanes1, lanes2):
        center_unequal = any(
            not np.array_equal(pts1, pts2)
            for pts1, pts2 in zip(lane1.points, lane2.points)
        )
        left_unequal = any(
            not np.array_equal(pts1, pts2)
            for pts1, pts2 in zip(lane1.left_points, lane2.left_points)
        )
        right_unequal = any(
            not np.array_equal(pts1, pts2)
            for pts1, pts2 in zip(lane1.right_points, lane2.right_points)
        )
        start_unequal = lane1.start != lane2.start
        end_unequal = lane1.end != lane2.end

        if (
            center_unequal
            or left_unequal
            or right_unequal
            or start_unequal
            or end_unequal
        ):
            return False

    return True


@pytest.mark.parametrize("params,check_intersections", GENERATOR_CASES)
def test_generator(params, check_intersections, tmp_path):
    rng = np.random.default_rng(0)
    lanes = generate_random_lanes(rng, params)

    lanes_serialized = serialize_lanes(lanes)
    lanes_unserialized = unserialize_lanes(lanes_serialized)

    assert lanelist_equality(lanes, lanes_unserialized)

    file_path = str(tmp_path / "data.npz")
    save_lanes_to_disk(file_path, lanes_unserialized)
    lanes_copy = load_lanes_from_disk(file_path)

    assert lanelist_equality(lanes, lanes_copy)

    if check_intersections:
        lane_to_grid, grid_to_lanes = lanes_spatial_hash(lanes, 100)
        get_all_intersection_points(lanes, lane_to_grid, grid_to_lanes)
