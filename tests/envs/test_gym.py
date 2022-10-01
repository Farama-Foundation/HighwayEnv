import gym
import pytest

import highway_env

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

