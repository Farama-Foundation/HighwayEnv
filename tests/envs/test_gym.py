import warnings

import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import highway_env
from highway_env.envs.common.abstract import MultiAgentWrapper
from highway_env.envs.highway_env import HighwayEnv


gym.register_envs(highway_env)


highway_env_ids = [
    env_id
    for env_id, env_spec in gym.registry.items()
    if isinstance(env_spec.entry_point, str) and "highway_env" in env_spec.entry_point
]


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is infinity. This is probably too high.",
        # "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
        "The environment intersection-v0 is out of date. You should consider upgrading to version `v1`.",
        "The environment intersection-multi-agent-v0 is out of date. You should consider upgrading to version `v1`.",
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
