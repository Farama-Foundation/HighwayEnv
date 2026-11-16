import gymnasium as gym
import pytest

import highway_env  # noqa: F401


class FakeVideoRecorder:
    """Stand-in for gymnasium's RecordVideo, which reads its frame rate once at creation."""

    def __init__(self, env):
        self.frames_per_sec = env.metadata.get("render_fps", 30)
        self.captured_frames = 0

    def _capture_frame(self):
        self.captured_frames += 1


def _make_env_with_recorder(env_id="highway-fast-v0", **config):
    env = gym.make(env_id, render_mode="rgb_array", config=config)
    recorder = FakeVideoRecorder(env)
    env.unwrapped.set_record_video_wrapper(recorder)
    return env, recorder


def test_video_frame_rate_default_speed():
    """By default, videos are played twice faster than real time."""
    env, recorder = _make_env_with_recorder()
    env.reset(seed=0)
    assert recorder.frames_per_sec == 2 * env.unwrapped.config["simulation_frequency"]
    env.close()


@pytest.mark.parametrize("ratio", [1, 0.5, 4])
def test_video_real_time_ratio(ratio):
    env, recorder = _make_env_with_recorder(video_real_time_ratio=ratio)
    env.reset(seed=0)
    assert recorder.frames_per_sec == pytest.approx(
        ratio * env.unwrapped.config["simulation_frequency"]
    )
    env.close()


def test_video_frame_rate_follows_configure():
    """The recorder frame rate is updated when the simulation frequency changes after it was set."""
    env, recorder = _make_env_with_recorder()
    env.unwrapped.configure({"simulation_frequency": 15, "video_real_time_ratio": 1})
    env.reset(seed=0)
    assert recorder.frames_per_sec == 15
    assert env.unwrapped.metadata["render_fps"] == 15

    # One frame is recorded per simulation step, so the video lasts as long as the simulation
    env.render()  # creates the viewer, which is required to capture intermediate frames
    recorder.captured_frames = 0
    env.step(env.unwrapped.action_type.actions_indexes["IDLE"])
    simulated_seconds = 1 / env.unwrapped.config["policy_frequency"]
    assert recorder.captured_frames / recorder.frames_per_sec == pytest.approx(
        simulated_seconds, abs=1 / 15
    )
    env.close()
