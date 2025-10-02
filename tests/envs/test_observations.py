import gymnasium as gym
import pytest

import highway_env


gym.register_envs(highway_env)


@pytest.mark.parametrize(
    "observation_config",
    [
        {"type": "LidarObservation"},
    ],
)
def test_observation_type(observation_config):
    env = gym.make("parking-v0", config={"observation": observation_config})
    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.action_space.contains(action)
        assert env.observation_space.contains(obs)
    env.close()


if __name__ == "__main__":
    pytest.main([__file__])
