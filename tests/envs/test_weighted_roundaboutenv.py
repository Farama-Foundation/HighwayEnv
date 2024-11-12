import pytest

from highway_env.envs import WeightedRoundaboutEnv


def test_weighted_roundabout_env_can_create():
    try:
        env = WeightedRoundaboutEnv()
    except:
        pytest.fail("Failed to create WeightedRoundaboutEnv")
