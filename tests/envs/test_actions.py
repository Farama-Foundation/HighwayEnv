import gymnasium as gym
import numpy as np
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


@pytest.mark.parametrize("lateral", [False, True])
@pytest.mark.parametrize(
    "speed_range, acceleration",
    [(None, 1.0), ((0, 20), 1.0), ((30, 40), -1.0)],
)
def test_discrete_action_speed_range(lateral, speed_range, acceleration):
    config = {
        "vehicles_count": 0,
        "action": {
            "lateral": lateral,
            "speed_range": speed_range,
        },
    }
    continuous_env = gym.make(
        "highway-v0",
        config={**config, "action": {**config["action"], "type": "ContinuousAction"}},
    )
    discrete_env = gym.make(
        "highway-v0",
        config={**config, "action": {**config["action"], "type": "DiscreteAction"}},
    )
    continuous_action = np.array([acceleration, 0.0] if lateral else [acceleration])
    discrete_action = (
        (7 if lateral else 2) if acceleration > 0 else (1 if lateral else 0)
    )
    try:
        continuous_env.reset(seed=0)
        discrete_env.reset(seed=0)
        for _ in range(3):
            expected_obs, expected_reward, _, _, expected_info = continuous_env.step(
                continuous_action
            )
            obs, reward, _, _, info = discrete_env.step(discrete_action)
            assert info["speed"] == pytest.approx(expected_info["speed"])
            assert reward == pytest.approx(expected_reward)
            np.testing.assert_allclose(obs, expected_obs)
    finally:
        continuous_env.close()
        discrete_env.close()
