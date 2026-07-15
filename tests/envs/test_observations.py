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


def test_occupancy_grid_shape_no_uint8_overflow():
    # Regression: grid_shape was cast to uint8 and silently wrapped
    # (mod 256) for grids with more than 255 cells along an axis.
    config = {
        "observation": {
            "type": "OccupancyGrid",
            "grid_size": [[-300, 300], [-10, 10]],
            "grid_step": [2, 2],
        }
    }
    env = gym.make("highway-v0", config=config)
    obs, _ = env.reset()
    # 4 default features (presence, vx, vy, on_road) x 300 x 10 cells
    assert env.observation_space.shape == (4, 300, 10)
    assert obs.shape == (4, 300, 10)
    env.close()


if __name__ == "__main__":
    pytest.main([__file__])
