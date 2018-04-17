from __future__ import division, print_function
import pytest

from highway_env.road.road import Road


def test_random():
    r = Road.create_random_road(lanes_count=6, lane_width=4.0, vehicles_count=42)
    assert len(r.lanes) == 6
    assert len(r.vehicles) == 42
