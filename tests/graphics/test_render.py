import gymnasium as gym
import numpy as np
import pytest

import highway_env


gym.register_envs(highway_env)


@pytest.mark.parametrize("env_spec", ["highway-v0", "merge-v0"])
def test_render(env_spec):
    env = gym.make(env_spec, render_mode="rgb_array").unwrapped
    env.reset()
    img = env.render()
    env.close()
    assert isinstance(img, np.ndarray)
    assert img.shape == (
        env.config["screen_height"],
        env.config["screen_width"],
        3,
    )  # (H,W,C)


@pytest.mark.parametrize("env_spec", ["highway-v0", "merge-v0"])
def test_obs_grayscale(env_spec, stack_size=4):
    env = gym.make(env_spec).unwrapped
    env.config.update(
        {
            "offscreen_rendering": True,
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (
                    env.config["screen_width"],
                    env.config["screen_height"],
                ),
                "stack_size": stack_size,
                "weights": [0.2989, 0.5870, 0.1140],
            },
        }
    )
    obs, info = env.reset()
    env.close()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (
        stack_size,
        env.config["screen_width"],
        env.config["screen_height"],
    )


@pytest.mark.parametrize("cells", [3, 16, 60, 61, 62, 123])
def test_render_lidar_observation(cells):
    """Rendering a lidar observation draws one sector per cell.

    Before v1.12.2, the angles were calculated by accumulating a float step,
    which for some cell counts would produced one angle too many and the drawing
    logic then result in an IndexError.
    """
    env = gym.make(
        "highway-v0",
        render_mode="rgb_array",
        config={"observation": {"type": "LidarObservation", "cells": cells}},
    ).unwrapped
    env.reset(seed=0)
    img = env.render()
    env.close()

    assert isinstance(img, np.ndarray)
    assert img.shape == (
        env.config["screen_height"],
        env.config["screen_width"],
        3,
    )
