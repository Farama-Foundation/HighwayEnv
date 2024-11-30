import pytest

from highway_env.envs import WeightedIntersectionEnv

def test_can_create():
    try:
        env = WeightedIntersectionEnv()
    except:
        pytest.fail("Could not create WeightedIntersectionEnv")