from __future__ import division, print_function

import importlib
from functools import partial
import numpy as np

from highway_env import utils


def finite_mdp(env):
    """
        Time-To-Collision (TTC) representation of the state.

        The state reward is defined from a occupancy grid over different TTCs and lanes. The grid cells encode the
        probability that the ego-vehicle will collide with another vehicle if it is located on a given lane in a given
        duration, under the hypothesis that every vehicles observed will maintain a constant velocity (including the
        ego-vehicle) and not change lane (excluding the ego-vehicle).

        For instance, in a three-lane road with a vehicle on the left lane with collision predicted in 5s the grid will
        be:
        [0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0]
        The TTC-state is a coordinate (lane, time) within this grid.

        If the ego-vehicle has the ability to change its velocity, an additional layer is added to the occupancy grid
        to iterate over the different velocity choices available.

        Finally, this state is flattened for compatibility with the FiniteMDP environment.
    """
    # Compute TTC grid
    grid = compute_ttc_grid(env)

    # Compute current state
    grid_state = (env.vehicle.speed_index(), env.vehicle.lane_index, 0)
    state = np.ravel_multi_index(grid_state, grid.shape)

    # Compute transition function
    transition_model_with_grid = partial(transition_model, grid=grid)
    transition = np.fromfunction(transition_model_with_grid, grid.shape + (env.action_space.n,), dtype=int)
    transition = np.reshape(transition, (np.size(grid), env.action_space.n))

    # Compute reward function
    v, l, t = grid.shape
    lanes = np.arange(l)/(l - 1)
    velocities = np.arange(v)/(v - 1)
    state_reward = \
        + env.COLLISION_REWARD * grid \
        + env.RIGHT_LANE_REWARD * np.tile(lanes[np.newaxis, :, np.newaxis], (v, 1, t)) \
        + env.HIGH_VELOCITY_REWARD * np.tile(velocities[:, np.newaxis, np.newaxis], (1, l, t))
    state_reward = np.ravel(state_reward)
    action_reward = [env.LANE_CHANGE_REWARD, 0, env.LANE_CHANGE_REWARD, 0, 0]
    reward = np.fromfunction(np.vectorize(lambda s, a: state_reward[s] + action_reward[a]),
                             (np.size(state_reward), np.size(action_reward)),  dtype=int)

    # Compute terminal states
    collision = grid == 1
    end_of_horizon = np.fromfunction(lambda h, i, j: j == grid.shape[2] - 1, grid.shape, dtype=int)
    terminal = collision | end_of_horizon

    # Creation of a new finite MDP
    try:
        module = importlib.import_module("finite_mdp.envs.finite_mdp")
        mdp = module.DeterministicMDP(transition, reward, terminal, state=state)
        return mdp
    except ModuleNotFoundError as e:
        raise "The finite_mdp module is required for conversion. {}".format(e)


def compute_ttc_grid(env):
    """
        For each ego-velocity and lane, compute the predicted time-to-collision to each vehicle within the lane and
        store the results in an occupancy grid.
    """
    time_quantization = 1.0  # The time quantization used in the state representation [s]
    horizon = 10.0
    grid = np.zeros((env.vehicle.SPEED_COUNT, len(env.road.lanes), int(horizon / time_quantization)))
    for velocity_index in range(grid.shape[0]):
        ego_velocity = env.vehicle.index_to_speed(velocity_index)
        for other in env.road.vehicles:
            if (other is env.vehicle) or (ego_velocity == other.velocity):
                continue
            margin = other.LENGTH / 2 + env.vehicle.LENGTH / 2
            collision_points = [(0, 1), (-margin, 0.5), (margin, 0.5)]
            for m, cost in collision_points:
                distance = env.vehicle.lane_distance_to(other) + m
                time_to_collision = distance / utils.not_zero(ego_velocity - other.velocity)
                if time_to_collision < 0:
                    continue
                # Quantize time-to-collision to both upper and lower values
                lane = other.lane_index
                for time in [int(time_to_collision / time_quantization),
                             int(np.ceil(time_to_collision / time_quantization))]:
                    if 0 <= lane < grid.shape[1] and 0 <= time < grid.shape[2]:
                        grid[velocity_index, lane, time] = max(grid[velocity_index, lane, time], cost)
    return grid


def transition_model(h, i, j, a, grid):
    """
        Deterministic transition from a position in the grid to the next.

    :param h: velocity index
    :param i: lane index
    :param j: time index
    :param a: action index
    :param grid: ttc grid specifying the limits of velocities, lanes, time and actions
    """
    # Idle action (1) as default transition
    next_state = clip_position(h, i, j + 1, grid)
    left = a == 0
    right = a == 2
    faster = (a == 3) & (j == 0)
    slower = (a == 4) & (j == 0)
    next_state[left] = clip_position(h[left], i[left] - 1, j[left] + 1, grid)
    next_state[right] = clip_position(h[right], i[right] + 1, j[right] + 1, grid)
    next_state[faster] = clip_position(h[faster] + 1, i[faster], j[faster] + 1, grid)
    next_state[slower] = clip_position(h[slower] - 1, i[slower], j[slower] + 1, grid)
    return next_state


def clip_position(h, i, j, grid):
    """
        Clip a position in the TTC grid, so that it stays within bounds.

    :param h: velocity index
    :param i: lane index
    :param j: time index
    :param grid: the ttc grid
    :return: The raveled index of the clipped position
    """
    h = np.clip(h, 0, grid.shape[0] - 1)
    i = np.clip(i, 0, grid.shape[1] - 1)
    j = np.clip(j, 0, grid.shape[2] - 1)
    indexes = np.ravel_multi_index((h, i, j), grid.shape)
    return indexes
