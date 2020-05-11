import pytest

from highway_env.road.lane import StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle


def test_network():
    # Diamond
    net = RoadNetwork()
    net.add_lane(0, 1, StraightLane([0, 0], [10, 0]))
    net.add_lane(1, 2, StraightLane([10, 0], [5, 5]))
    net.add_lane(2, 0, StraightLane([5, 5], [0, 0]))
    net.add_lane(1, 3, StraightLane([10, 0], [5, -5]))
    net.add_lane(3, 0, StraightLane([5, -5], [0, 0]))
    print(net.graph)

    # Road
    road = Road(network=net)
    v = ControlledVehicle(road, [5, 0], heading=0, target_speed=2)
    road.vehicles.append(v)
    assert v.lane_index == (0, 1, 0)

    # Lane changes
    dt = 1/15
    lane_index = v.target_lane_index
    lane_changes = 0
    for _ in range(int(20/dt)):
        road.act()
        road.step(dt)
        if lane_index != v.target_lane_index:
            lane_index = v.target_lane_index
            lane_changes += 1
    assert lane_changes >= 3
