import gym
import numpy as np
import pytest

import highway_env

envs = ["highway-v0", "merge-v0"]


@pytest.mark.parametrize("env_spec", envs)
def test_render(env_spec):
    env = gym.make(env_spec)
    env.configure({"offscreen_rendering": True})
    img = env.render(mode="rgb_array")
    env.close()
    assert isinstance(img, np.ndarray)
    assert img.shape == (env.config["screen_height"], env.config["screen_width"], 3)  # (H,W,C)


@pytest.mark.parametrize("env_spec", envs)
def test_obs_grayscale(env_spec, stack_size=4):
    env = gym.make(env_spec)
    env.configure({
        "offscreen_rendering": True,
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (env.config["screen_width"], env.config["screen_height"]),
            "stack_size": stack_size,
            "weights": [0.2989, 0.5870, 0.1140],
        }
    })
    obs = env.reset()
    env.close()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (stack_size, env.config["screen_width"], env.config["screen_height"])
