import numpy as np

from highway_env.road.generation.engine.gen_utils import Lane
from highway_env.road.generation.engine.validation import get_all_intersection_points
from highway_env.road.generation.generator import (
    generate_random_lanes,
    load_lanes_from_disk,
    print_lanes,
    save_lanes_to_disk,
    serialize_lanes,
    unserialize_lanes,
)
from highway_env.road.generation.spatial_hash import lanes_spatial_hash


params_list = [
    None,  # default
    {  # small
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
        "seed": None,
    },
    {  # big + chaotic
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
        "seed": 1,
    },
    {  # fast
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
        "seed": None,
    },
    {  # slow
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
        "seed": None,
    },
    {  # wide
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
        "seed": None,
    },
    {  # skinny
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
        "seed": None,
    },
    {  # jittery
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
        "seed": None,
    },
    {  # smooth
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
        "seed": None,
    },
    {  # twisty
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
        "seed": None,
    },
    {  # straight
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
        "seed": None,
    },
    {  # hyperdense
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
        "seed": None,
    },
    {  # sparse
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
        "seed": None,
    },
    {  # deadly
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
        "seed": None,
    },
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


file_path = "data.npz"


def test_generator():
    for i, params in enumerate(params_list):
        lanes = generate_random_lanes(params)

        lanes_serialized = serialize_lanes(lanes)
        lanes_unserialized = unserialize_lanes(lanes_serialized)

        assert lanelist_equality(lanes, lanes_unserialized)

        save_lanes_to_disk(file_path, lanes_unserialized)
        lanes_copy = load_lanes_from_disk(file_path)

        assert lanelist_equality(lanes, lanes_copy)

        if i == 1:
            print_lanes(lanes)
            lane_to_grid, grid_to_lanes = lanes_spatial_hash(lanes, 100)
            get_all_intersection_points(lanes, lane_to_grid, grid_to_lanes)
