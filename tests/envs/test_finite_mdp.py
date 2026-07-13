from functools import partial

import gymnasium as gym
import numpy as np

from highway_env.envs.common.finite_mdp import (
    clip_position,
    compute_ttc_grid,
    transition_model,
)


def test_clip_position_clips_to_grid_bounds():
    grid = np.zeros((3, 4, 5))
    index = clip_position(10, -2, 20, grid)
    h, i, j = np.unravel_index(index, grid.shape)
    assert (h, i, j) == (2, 0, 4)


def test_transition_model_via_fromfunction():
    grid = np.zeros((3, 3, 4))
    transition_model_with_grid = partial(transition_model, grid=grid)
    transition = np.fromfunction(
        transition_model_with_grid, grid.shape + (5,), dtype=int
    )
    assert transition.shape == grid.shape + (5,)
    assert transition[1, 1, 0, 1] == clip_position(1, 1, 1, grid)
    assert transition[1, 1, 0, 0] == clip_position(1, 0, 1, grid)
    assert transition[1, 1, 0, 2] == clip_position(1, 2, 1, grid)
    assert transition[1, 1, 0, 3] == clip_position(2, 1, 1, grid)
    assert transition[1, 1, 0, 4] == clip_position(0, 1, 1, grid)


def test_compute_ttc_grid_on_highway_env():
    env = gym.make("highway-v0")
    env.reset()
    unwrapped = env.unwrapped
    grid = compute_ttc_grid(unwrapped, time_quantization=1.0, horizon=5.0)
    road_lanes = unwrapped.road.network.all_side_lanes(unwrapped.vehicle.lane_index)
    assert grid.shape == (
        unwrapped.vehicle.target_speeds.size,
        len(road_lanes),
        5,
    )
    assert grid.min() >= 0
    env.close()


def test_compute_ttc_grid_predicts_collision():
    env = gym.make("highway-v0", config={"vehicles_count": 5})
    env.reset()
    unwrapped = env.unwrapped
    ego = unwrapped.vehicle
    other = next(v for v in unwrapped.road.vehicles if v is not ego)
    other.position = ego.position + ego.direction * 20
    other.speed = ego.speed - 5
    grid = compute_ttc_grid(unwrapped, time_quantization=1.0, horizon=10.0)
    assert grid.max() > 0
    env.close()
