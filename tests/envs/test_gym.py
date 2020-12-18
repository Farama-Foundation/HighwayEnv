import gym
import pytest

import highway_env

envs = [
    "highway-v0",
    "merge-v0",
    "roundabout-v0",
    "intersection-v0",
    "parking-v0",
    "summon-v0",
    "two-way-v0",
    "lane-keeping-v0",
]


@pytest.mark.parametrize("env_spec", envs)
def test_env_step(env_spec):
    env = gym.make(env_spec)

    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)

