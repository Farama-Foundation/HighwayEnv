import gymnasium as gym
import pytest

import highway_env


gym.register_envs(highway_env)

IDLE = 1
EXIT_LANE = ("1", "2", 6)
EXIT_RAMP = ("2", "exit", 0)


def place_ego(env, lane_index, longitudinal):
    """Move the ego-vehicle onto a lane, aligned with it, and target that lane."""
    lane = env.road.network.get_lane(lane_index)
    ego = env.vehicle
    ego.position = lane.position(longitudinal, 0)
    ego.heading = lane.heading_at(longitudinal)
    ego.target_lane_index = lane_index
    ego.on_state_update()
    assert ego.lane_index == lane_index


@pytest.fixture
def env_v2():
    env = gym.make("exit-v2")
    env.reset(seed=0)
    yield env.unwrapped
    env.close()


def test_exit_v2_default_config(env_v2):
    config = env_v2.config
    assert config["neighbour_vehicles_connected_lanes"] is True
    assert config["collision_reward"] < 0
    assert config["goal_reward"] > 0
    assert config["normalize_reward"] is False


def test_exit_v2_success_requires_taking_the_exit(env_v2):
    """Being on (or targeting) the exit lane is not enough: the ego-vehicle must be on the ramp."""
    place_ego(env_v2, EXIT_LANE, 50)
    assert not env_v2._is_success()
    assert not env_v2._is_terminated()

    place_ego(env_v2, EXIT_RAMP, 20)
    assert env_v2._is_success()
    assert env_v2._is_terminated()


def test_exit_v1_success_unchanged():
    """exit-v1 keeps counting the exit lane as success, for reproducibility."""
    env = gym.make("exit-v1")
    env.reset(seed=0)
    try:
        place_ego(env.unwrapped, EXIT_LANE, 50)
        assert env.unwrapped._is_success()
    finally:
        env.close()


def test_exit_v2_driving_past_the_exit_ends_the_episode_as_a_miss(env_v2):
    env_v2._previous_lane_progress = env_v2._lane_progress()
    place_ego(env_v2, ("2", "3", 3), 10)
    assert env_v2._is_terminated()
    assert not env_v2._is_success()
    assert env_v2._rewards(IDLE)["missed_exit_reward"] == 1


def test_exit_v2_running_out_of_time_counts_as_a_miss(env_v2):
    env_v2._previous_lane_progress = env_v2._lane_progress()
    assert env_v2._rewards(IDLE)["missed_exit_reward"] == 0
    env_v2.time = env_v2.config["duration"]
    assert env_v2._is_truncated()
    assert env_v2._rewards(IDLE)["missed_exit_reward"] == 1


def test_exit_v2_episode_ends_with_goal_reward_on_the_ramp(env_v2):
    env_v2.road.vehicles = [env_v2.vehicle]
    place_ego(env_v2, EXIT_LANE, 90)

    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env_v2.step(IDLE)

    assert terminated
    assert info["is_success"]
    assert info["rewards"]["goal_reward"] == 1
    assert reward >= env_v2.config["goal_reward"]


def test_exit_v2_collision_penalty(env_v2):
    env_v2._previous_lane_progress = env_v2._lane_progress()
    env_v2.vehicle.crashed = True
    assert env_v2._rewards(IDLE)["collision_reward"] == 1
    assert env_v2._rewards(IDLE)["missed_exit_reward"] == 0
    assert (
        env_v2._reward(IDLE)
        <= env_v2.config["collision_reward"] + env_v2.config["high_speed_reward"]
    )
    assert env_v2._is_terminated()


def test_exit_v2_lane_progress_is_a_potential_difference(env_v2):
    """Moving right earns the shaping reward, moving back left pays it back, and repeated calls agree."""
    lanes_count = env_v2.config["lanes_count"]
    place_ego(env_v2, ("0", "1", 2), 200)
    env_v2._previous_lane_progress = env_v2._lane_progress()

    place_ego(env_v2, ("0", "1", 3), 200)
    first = env_v2._rewards(IDLE)["lane_progress_reward"]
    assert first == pytest.approx(1 / lanes_count)
    assert env_v2._rewards(IDLE)["lane_progress_reward"] == first

    env_v2._previous_lane_progress = env_v2._lane_progress()
    place_ego(env_v2, ("0", "1", 2), 200)
    assert env_v2._rewards(IDLE)["lane_progress_reward"] == pytest.approx(
        -1 / lanes_count
    )


def test_exit_v2_lane_progress_reaches_one_on_the_ramp(env_v2):
    place_ego(env_v2, EXIT_RAMP, 20)
    assert env_v2._lane_progress() == 1.0
    place_ego(env_v2, EXIT_LANE, 50)
    assert env_v2._lane_progress() == 1.0
