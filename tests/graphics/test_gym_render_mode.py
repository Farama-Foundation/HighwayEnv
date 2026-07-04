import os
import warnings

import gymnasium as gym
import numpy as np
import pytest

import highway_env


gym.register_envs(highway_env)

ENV_ID = "highway-v0"

skip_headless = pytest.mark.skipif(
    os.environ.get("SDL_VIDEODRIVER") == "dummy",
    reason="Requires a display (skipped in headless CI)",
)


class TestRenderModeDefaults:
    def test_none_derives_offscreen(self):
        env = gym.make(ENV_ID)
        assert env.unwrapped.config["offscreen_rendering"] is True
        assert env.unwrapped.config["real_time_rendering"] is False
        env.close()

    def test_rgb_array_derives_offscreen(self):
        env = gym.make(ENV_ID, render_mode="rgb_array")
        assert env.unwrapped.config["offscreen_rendering"] is True
        assert env.unwrapped.config["real_time_rendering"] is False
        env.close()

    @skip_headless
    def test_human_derives_onscreen(self):
        env = gym.make(ENV_ID, render_mode="human")
        assert env.unwrapped.config["offscreen_rendering"] is False
        assert env.unwrapped.config["real_time_rendering"] is False
        env.close()

    def test_explicit_override_respected(self):
        env = gym.make(
            ENV_ID, render_mode="rgb_array", config={"offscreen_rendering": False}
        )
        assert env.unwrapped.config["offscreen_rendering"] is False
        env.close()


class TestRenderReturn:
    def test_none_returns_none(self):
        env = gym.make(ENV_ID)
        env.reset()
        result = env.render()
        assert result is None
        env.close()

    def test_rgb_array_returns_ndarray(self):
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env.reset()
        img = env.render()
        assert isinstance(img, np.ndarray)
        assert img.shape == (
            env.unwrapped.config["screen_height"],
            env.unwrapped.config["screen_width"],
            3,
        )
        env.close()

    @skip_headless
    def test_human_returns_none(self):
        env = gym.make(ENV_ID, render_mode="human")
        env.reset()
        result = env.render()
        assert result is None
        env.close()

    @skip_headless
    def test_rgb_array_with_window_still_returns_ndarray(self):
        env = gym.make(
            ENV_ID, render_mode="rgb_array", config={"offscreen_rendering": False}
        )
        env.reset()
        img = env.render()
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3
        env.close()

    def test_human_with_offscreen_returns_none(self):
        env = gym.make(
            ENV_ID, render_mode="human", config={"offscreen_rendering": True}
        )
        env.reset()
        result = env.render()
        assert result is None
        env.close()


class TestDeprecation:
    def test_offscreen_env_var_emits_warning(self):
        original_val = os.getenv("OFFSCREEN_RENDERING")
        os.environ["OFFSCREEN_RENDERING"] = "1"
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                env = gym.make(ENV_ID, render_mode="rgb_array")
                env.close()
            assert any(
                issubclass(x.category, DeprecationWarning)
                and "OFFSCREEN_RENDERING" in str(x.message)
                for x in w
            )
        finally:
            if original_val is None:
                del os.environ["OFFSCREEN_RENDERING"]
            else:
                os.environ["OFFSCREEN_RENDERING"] = original_val
