import pytest

from highway_env.envs import WeightedIntersectionEnv, WeightedHighwayEnv, CarpetCity, WeightedRoundaboutEnv, Stovring

def test_can_create():
    try:
        env = WeightedIntersectionEnv()
    except:
        pytest.fail("Could not create WeightedIntersectionEnv")

def test_weighted_roundabout_env_can_create():
    try:
        env = WeightedRoundaboutEnv()
    except:
        pytest.fail("Failed to create WeightedRoundaboutEnv")


def test_weighted_highway_env_can_create():
    try:
        env = WeightedHighwayEnv()
    except:
        pytest.fail("Failed to create WeightedHighwayEnv")

def test_dustring_env_can_create():
    try:
        env = Stovring()
    except:
        pytest.fail("Failed to create Stovring")

def test_carpet_city_can_create():
    try:
        env = CarpetCity()
    except:
        pytest.fail("Failed to create CarpetCity")