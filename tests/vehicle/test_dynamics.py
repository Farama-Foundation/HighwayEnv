import pytest

from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle, Landmark

FPS = 15


def test_step():
    v = Vehicle(road=None, position=[0, 0], speed=20, heading=0)
    for _ in range(2*FPS):
        v.step(dt=1/FPS)
    assert v.position[0] == pytest.approx(40)
    assert v.position[1] == pytest.approx(0)
    assert v.speed == pytest.approx(20)
    assert v.heading == pytest.approx(0)


def test_act():
    v = Vehicle(road=None, position=[0, 0], speed=20, heading=0)
    v.act({'acceleration': 1, 'steering': 0})
    for _ in range(1 * FPS):
        v.step(dt=1/FPS)
    assert v.speed == pytest.approx(21)

    v.act({'acceleration': 0, 'steering': 0.5})
    for _ in range(1 * FPS):
        v.step(dt=1/FPS)
    assert v.speed == pytest.approx(21)
    assert v.position[1] > 0


def test_brake():
    v = Vehicle(road=None, position=[0, 0], speed=20, heading=0)
    for _ in range(10 * FPS):
        v.act({'acceleration': min(max(-1 * v.speed, -6), 6), 'steering': 0})
        v.step(dt=1/FPS)
    assert v.speed == pytest.approx(0, abs=0.01)


def test_front():
    r = Road(RoadNetwork.straight_road_network(1))
    v1 = Vehicle(road=r, position=[0, 0], speed=20)
    v2 = Vehicle(road=r, position=[10, 0], speed=10)
    r.vehicles.extend([v1, v2])

    assert v1.lane_distance_to(v2) == pytest.approx(10)
    assert v2.lane_distance_to(v1) == pytest.approx(-10)


def test_collision():
    # Collision between two vehicles
    r = Road(RoadNetwork.straight_road_network(1))
    v1 = Vehicle(road=r, position=[0, 0], speed=10)
    v2 = Vehicle(road=r, position=[4, 0], speed=20)
    v1.handle_collisions(v2)

    assert v1.crashed and v2.crashed
    # Collision between a vehicle and an obstacle
    v3 = Vehicle(road=r, position=[20, 0], speed=10)
    o = Obstacle(road=r, position=[23, 0])
    v3.handle_collisions(o)

    assert v3.crashed and o.crashed
    # Collision between a vehicle and a landmark
    v4 = Vehicle(road=r, position=[40, 0], speed=10)
    l = Landmark(road=r, position=[43, 0])
    v4.handle_collisions(l)

    assert v4.crashed is False
    assert l.hit
