import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import highway_env
from highway_env.envs.common.abstract import MultiAgentWrapper
from highway_env.envs.highway_env import HighwayEnv


gym.register_envs(highway_env)


highway_env_ids = [
    env_id
    for env_id, env_spec in gym.registry.items()
    if isinstance(env_spec.entry_point, str)
    and "highway_env" in env_spec.entry_point
    and "intersection_pz_env" not in env_spec.entry_point
]


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is infinity. This is probably too high.",
        # "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
        "The environment exit-v0 is out of date. You should consider upgrading to version `v1`.",
        "The environment merge-v0 is out of date. You should consider upgrading to version `v1`.",
        "The environment merge-generic-v0 is out of date. You should consider upgrading to version `v1`.",
        "The environment racetrack-v0 is out of date. You should consider upgrading to version `v1`.",
        "The environment racetrack-large-v0 is out of date. You should consider upgrading to version `v1`.",
        "The environment racetrack-oval-v0 is out of date. You should consider upgrading to version `v1`.",
        "The environment roundabout-v0 is out of date. You should consider upgrading to version `v1`.",
        "The environment roundabout-generic-v0 is out of date. You should consider upgrading to version `v1`.",
        "The environment u-turn-v0 is out of date. You should consider upgrading to version `v1`.",
        "The environment intersection-v0 is out of date. You should consider upgrading to version `v2`.",
        "The environment intersection-v1 is out of date. You should consider upgrading to version `v2`.",
        "The environment intersection-multi-agent-v0 is out of date. You should consider upgrading to version `v2`.",
        "The environment intersection-multi-agent-v1 is out of date. You should consider upgrading to version `v2`.",
    ]
]


@pytest.mark.parametrize("env_id", highway_env_ids)
def test_highway_api(env_id):
    """Check that all environments pass the environment checker with no warnings other than the expected."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = gym.make(env_id)
        if isinstance(env, MultiAgentWrapper):
            pytest.skip(f"Multi-Agent wrapper does not match api ({env}).")

        check_env(env, skip_render_check=True)

        env.close()

    for warning in caught_warnings:
        if "is different from the unwrapped version" in warning.message.args[0]:
            continue
        elif warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise gym.error.Error(f"Unexpected warning: {warning.message}")


@pytest.mark.parametrize(
    "env_spec",
    [
        "highway-v0",
        "merge-v0",
        "roundabout-v0",
        "intersection-v0",
        "intersection-v1",
        "parking-v0",
        "two-way-v0",
        "lane-keeping-v0",
        "racetrack-v0",
    ],
)
def test_env_step(env_spec):
    env = gym.make(env_spec)

    obs, info = env.reset()
    assert env.observation_space.contains(obs)

    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
    env.close()


def test_env_reset_options(env_spec: str = "highway-v0"):
    env = gym.make(env_spec).unwrapped

    default_duration = HighwayEnv().default_config()["duration"]
    assert env.config["duration"] == default_duration

    update_duration = default_duration * 2
    env.reset(options={"config": {"duration": update_duration}})
    assert env.config["duration"] == update_duration


@pytest.mark.parametrize(
    ("old_env_spec", "new_env_spec"),
    [
        ("exit-v0", "exit-v1"),
        ("merge-v0", "merge-v1"),
        ("merge-generic-v0", "merge-generic-v1"),
        ("roundabout-v0", "roundabout-v1"),
        ("roundabout-generic-v0", "roundabout-generic-v1"),
        ("racetrack-v0", "racetrack-v1"),
        ("racetrack-large-v0", "racetrack-large-v1"),
        ("racetrack-oval-v0", "racetrack-oval-v1"),
        ("u-turn-v0", "u-turn-v1"),
        ("intersection-v0", "intersection-v2"),
        ("intersection-multi-agent-v0", "intersection-multi-agent-v2"),
    ],
)
def test_connected_lane_neighbour_versions(old_env_spec, new_env_spec):
    old_env = gym.make(old_env_spec)
    new_env = gym.make(new_env_spec, config={"duration": 1})

    try:
        old_unwrapped = old_env.unwrapped
        new_unwrapped = new_env.unwrapped

        assert old_unwrapped.config["neighbour_vehicles_connected_lanes"] is False
        assert old_unwrapped.road.neighbour_vehicles_connected_lanes is False
        assert new_unwrapped.config["neighbour_vehicles_connected_lanes"] is True
        assert new_unwrapped.road.neighbour_vehicles_connected_lanes is True
        assert new_unwrapped.config["duration"] == 1
    finally:
        old_env.close()
        new_env.close()


@pytest.mark.parametrize(
    "env_spec",
    [
        "highway-v0",
        "merge-v0",
        "roundabout-v0",
        "intersection-v0",
        "intersection-v1",
        "parking-v0",
        "two-way-v0",
        "racetrack-v0",
    ],
)
def test_env_vectorization__info_dtype_is_float(env_spec):
    def thunk(**config_kwargs):
        def make():
            return gym.make(env_spec, config=config_kwargs)

        return make

    envs = gym.vector.SyncVectorEnv(
        [
            thunk(duration=2, simulation_frequency=2),
            thunk(duration=1, simulation_frequency=2),
        ],
        autoreset_mode="SameStep",
    )

    _obs, info = envs.reset()
    assert np.issubdtype(info["speed"].dtype, np.floating)

    zero_action = np.zeros(envs.action_space.shape, envs.action_space.dtype)
    # run until first environment with longer duration terminates
    for _step in range(2):
        _obs, _reward, _terminated, truncated, info = envs.step(zero_action)
        assert np.issubdtype(info["speed"].dtype, np.floating)
        if truncated[0]:
            break

    envs.close()
