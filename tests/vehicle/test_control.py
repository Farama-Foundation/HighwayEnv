import pytest

from highway_env.road.lane import StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle

FPS = 15


def test_step():
    v = ControlledVehicle(road=None, position=[0, 0], speed=20, heading=0)
    for _ in range(2 * FPS):
        v.step(dt=1/FPS)
    assert v.position[0] == pytest.approx(40)
    assert v.position[1] == pytest.approx(0)
    assert v.speed == pytest.approx(20)
    assert v.heading == pytest.approx(0)


def test_lane_change():
    road = Road(RoadNetwork.straight_road_network(2))
    v = ControlledVehicle(road=road, position=road.network.get_lane(("0", "1", 0)).position(0, 0), speed=20, heading=0)
    v.act('LANE_RIGHT')
    for _ in range(3 * FPS):
        v.act()
        v.step(dt=1/FPS)
    assert v.speed == pytest.approx(20)
    assert v.position[1] == pytest.approx(StraightLane.DEFAULT_WIDTH, abs=StraightLane.DEFAULT_WIDTH/4)
    assert v.lane_index[2] == 1


def test_speed_control():
    road = Road(RoadNetwork.straight_road_network(1))
    v = ControlledVehicle(road=road, position=road.network.get_lane(("0", "1", 0)).position(0, 0), speed=20, heading=0)
    v.act('FASTER')
    for _ in range(int(3 * v.TAU_ACC * FPS)):
        v.act()
        v.step(dt=1/FPS)
    assert v.speed == pytest.approx(20 + v.DELTA_SPEED, abs=0.5)
    assert v.position[1] == pytest.approx(0)
    assert v.lane_index[2] == 0
