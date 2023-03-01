import gymnasium as gym
import pytest
import highway_env
from highway_env.envs.highway_env import HighwayEnv

highway_env.register_highway_envs()

envs = [
    "highway-v0",
    "merge-v0",
    "roundabout-v0",
    "intersection-v0",
    "intersection-v1",
    "parking-v0",
    "two-way-v0",
    "lane-keeping-v0",
    "racetrack-v0",
]


@pytest.mark.parametrize("env_spec", envs)
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
    env = gym.make(env_spec)

    default_duration = HighwayEnv().default_config()["duration"]
    assert env.config["duration"] == default_duration

    update_duration = default_duration * 2
    env.reset(options={"config": {"duration": update_duration}})
    assert env.config["duration"] == update_duration
