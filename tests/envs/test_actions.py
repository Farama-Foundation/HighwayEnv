import gymnasium as gym
import pytest

import highway_env


gym.register_envs(highway_env)


@pytest.mark.parametrize(
    "action_config",
    [
        {"type": "ContinuousAction"},
        {"type": "DiscreteAction"},
        {"type": "DiscreteMetaAction"},
    ],
)
def test_action_type(action_config):
    env = gym.make("highway-v0", config={"action": action_config})
    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.action_space.contains(action)
        assert env.observation_space.contains(obs)
    env.close()
