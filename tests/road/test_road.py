import numpy as np
import pytest

from highway_env.road.lane import StraightLane, CircularLane, PolyLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle


@pytest.fixture
def net() -> RoadNetwork:
    # Diamond
    net = RoadNetwork()
    net.add_lane(0, 1, StraightLane([0, 0], [10, 0]))
    net.add_lane(1, 2, StraightLane([10, 0], [5, 5]))
    net.add_lane(2, 0, StraightLane([5, 5], [0, 0]))
    net.add_lane(1, 3, StraightLane([10, 0], [5, -5]))
    net.add_lane(3, 0, StraightLane([5, -5], [0, 0]))
    print(net.graph)

    return net


def test_network(net):
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


def test_network_to_from_config(net):
    config_dict = net.to_config()
    net_2 = RoadNetwork.from_config(config_dict)
    assert len(net.graph) == len(net_2.graph)


def test_polylane():
    lane = CircularLane(
        center=[0, 0],
        radius=10,
        start_phase=0,
        end_phase=3.14,
    )

    num_samples = int(lane.length / 5)
    sampled_centreline = [
        lane.position(longitudinal=lon, lateral=0)
        for lon in np.linspace(0, lane.length, num_samples)
    ]
    sampled_left_boundary = [
        lane.position(longitudinal=lon, lateral=0.5 * lane.width_at(longitudinal=lon))
        for lon in np.linspace(0, lane.length, num_samples)
    ]
    sampled_right_boundary = [
        lane.position(longitudinal=lon, lateral=-0.5 * lane.width_at(longitudinal=lon))
        for lon in np.linspace(0, lane.length, num_samples)
    ]
    polylane = PolyLane(
        lane_points=sampled_centreline,
        left_boundary_points=sampled_left_boundary,
        right_boundary_points=sampled_right_boundary,
    )

    # sample boundaries from both lanes and assert equal

    num_samples = int(lane.length / 3)
    # original lane
    sampled_centreline = [
        lane.position(longitudinal=lon, lateral=0)
        for lon in np.linspace(0, lane.length, num_samples)
    ]
    sampled_left_boundary = [
        lane.position(longitudinal=lon, lateral=0.5 * lane.width_at(longitudinal=lon))
        for lon in np.linspace(0, lane.length, num_samples)
    ]
    sampled_right_boundary = [
        lane.position(longitudinal=lon, lateral=-0.5 * lane.width_at(longitudinal=lon))
        for lon in np.linspace(0, lane.length, num_samples)
    ]

    # polylane
    polylane_sampled_centreline = [
        polylane.position(longitudinal=lon, lateral=0)
        for lon in np.linspace(0, polylane.length, num_samples)
    ]
    polylane_sampled_left_boundary = [
        polylane.position(
            longitudinal=lon, lateral=0.5 * polylane.width_at(longitudinal=lon)
        )
        for lon in np.linspace(0, polylane.length, num_samples)
    ]
    polylane_sampled_right_boundary = [
        polylane.position(
            longitudinal=lon, lateral=-0.5 * polylane.width_at(longitudinal=lon)
        )
        for lon in np.linspace(0, polylane.length, num_samples)
    ]

    # assert equal (very coarse because of coarse sampling)
    assert all(
        np.linalg.norm(
            np.array(sampled_centreline) - np.array(polylane_sampled_centreline), axis=1
        )
        < 0.7
    )
    assert all(
        np.linalg.norm(
            np.array(sampled_left_boundary) - np.array(polylane_sampled_left_boundary),
            axis=1,
        )
        < 0.7
    )
    assert all(
        np.linalg.norm(
            np.array(sampled_right_boundary)
            - np.array(polylane_sampled_right_boundary),
            axis=1,
        )
        < 0.7
    )
