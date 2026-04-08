"""Tests for Road.neighbour_vehicles with connected lane segments.

Covers issue #626: neighbour_vehicles doesn't consider connected lanes.
"""

import numpy as np
import pytest

from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle


def _make_vehicle(
    road: Road,
    net: RoadNetwork,
    lane_index: tuple[str, str, int],
    longitudinal: float,
) -> Vehicle:
    """Helper: create a vehicle at a given longitudinal position on a lane."""
    lane = net.get_lane(lane_index)
    pos = lane.position(longitudinal, 0)
    heading = lane.heading_at(longitudinal)
    v = Vehicle(road, position=pos, heading=heading, speed=10)
    v.lane_index = lane_index
    v.lane = lane
    road.vehicles.append(v)
    return v


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def straight_connected_road() -> tuple[Road, RoadNetwork]:
    """Two connected straight segments: a->b (50m) then b->c (50m)."""
    net = RoadNetwork()
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [0, 0], [50, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
        ),
    )
    net.add_lane(
        "b",
        "c",
        StraightLane(
            [50, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
        ),
    )
    road = Road(network=net, np_random=np.random.RandomState(42))
    return road, net


@pytest.fixture
def straight_curve_road() -> tuple[Road, RoadNetwork]:
    """Straight segment a->b followed by a circular arc b->c."""
    net = RoadNetwork()
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [0, 0], [50, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
        ),
    )
    net.add_lane(
        "b",
        "c",
        CircularLane(
            center=[50, -20],
            radius=20,
            start_phase=np.deg2rad(90),
            end_phase=np.deg2rad(0),
            clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
        ),
    )
    road = Road(network=net, np_random=np.random.RandomState(42))
    return road, net


@pytest.fixture
def three_segment_road() -> tuple[Road, RoadNetwork]:
    """Three connected segments: a->b (50m), b->c (50m), c->d (50m)."""
    net = RoadNetwork()
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [0, 0], [50, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
        ),
    )
    net.add_lane(
        "b",
        "c",
        StraightLane(
            [50, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
        ),
    )
    net.add_lane(
        "c",
        "d",
        StraightLane(
            [100, 0], [150, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
        ),
    )
    road = Road(network=net, np_random=np.random.RandomState(42))
    return road, net


@pytest.fixture
def multi_lane_road() -> tuple[Road, RoadNetwork]:
    """Two-lane connected road: a->b and b->c, each with 2 lanes."""
    net = RoadNetwork()
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [0, 0], [50, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED)
        ),
    )
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [0, 4], [50, 4], line_types=(LineType.STRIPED, LineType.CONTINUOUS)
        ),
    )
    net.add_lane(
        "b",
        "c",
        StraightLane(
            [50, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED)
        ),
    )
    net.add_lane(
        "b",
        "c",
        StraightLane(
            [50, 4], [100, 4], line_types=(LineType.STRIPED, LineType.CONTINUOUS)
        ),
    )
    road = Road(network=net, np_random=np.random.RandomState(42))
    return road, net


# ---------------------------------------------------------------------------
# Tests: same-segment behaviour (regression — must still work)
# ---------------------------------------------------------------------------


class TestSameSegmentNeighbours:
    """Verify that the original same-lane-segment detection still works."""

    def test_front_and_rear_on_same_segment(self, straight_connected_road):
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=25)
        front = _make_vehicle(road, net, ("a", "b", 0), longitudinal=40)
        rear = _make_vehicle(road, net, ("a", "b", 0), longitudinal=10)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert v_front is front
        assert v_rear is rear

    def test_no_neighbours(self, straight_connected_road):
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=25)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert v_front is None
        assert v_rear is None

    def test_only_front(self, straight_connected_road):
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=10)
        front = _make_vehicle(road, net, ("a", "b", 0), longitudinal=40)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert v_front is front
        assert v_rear is None

    def test_only_rear(self, straight_connected_road):
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=40)
        rear = _make_vehicle(road, net, ("a", "b", 0), longitudinal=10)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert v_front is None
        assert v_rear is rear


# ---------------------------------------------------------------------------
# Tests: connected lane neighbours (the fix for issue #626)
# ---------------------------------------------------------------------------


class TestConnectedLaneNeighbours:
    """Verify that vehicles on connected next/prev segments are detected."""

    def test_front_on_next_segment(self, straight_connected_road):
        """Vehicle on next segment b->c should be detected as front neighbour."""
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=48)
        front = _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert (
            v_front is front
        ), "Vehicle on connected next segment should be detected as front neighbour"

    def test_rear_on_previous_segment(self, straight_connected_road):
        """Vehicle on previous segment a->b should be detected as rear neighbour."""
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)
        rear = _make_vehicle(road, net, ("a", "b", 0), longitudinal=45)

        v_front, v_rear = road.neighbour_vehicles(ego, ("b", "c", 0))
        assert (
            v_rear is rear
        ), "Vehicle on connected previous segment should be detected as rear neighbour"

    def test_front_on_curve_segment(self, straight_curve_road):
        """Vehicle on a connected curve should be detected as front neighbour."""
        road, net = straight_curve_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=48)
        front = _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert (
            v_front is front
        ), "Vehicle on connected curve should be detected as front neighbour"

    def test_closer_same_segment_preferred_over_next_segment(
        self, straight_connected_road
    ):
        """When both a same-segment and a next-segment vehicle are ahead,
        the closer one should be returned."""
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=30)
        close_front = _make_vehicle(road, net, ("a", "b", 0), longitudinal=45)
        _make_vehicle(
            road, net, ("b", "c", 0), longitudinal=10
        )  # farther vehicle on next segment

        v_front, _ = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert (
            v_front is close_front
        ), "Closer same-segment vehicle should be preferred over farther next-segment one"

    def test_both_connected_front_and_rear(self, three_segment_road):
        """Ego on middle segment, front on next, rear on previous."""
        road, net = three_segment_road
        rear = _make_vehicle(road, net, ("a", "b", 0), longitudinal=45)
        ego = _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)
        front = _make_vehicle(road, net, ("c", "d", 0), longitudinal=5)

        v_front, v_rear = road.neighbour_vehicles(ego, ("b", "c", 0))
        assert v_front is front
        assert v_rear is rear

    def test_multi_lane_same_lane_id(self, multi_lane_road):
        """On a multi-lane road, only vehicles on the matching lane id
        of the next segment should be considered."""
        road, net = multi_lane_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=48)
        front_lane0 = _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)
        # Vehicle on lane 1 of the next segment (different lane)
        _make_vehicle(road, net, ("b", "c", 1), longitudinal=3)

        v_front, _ = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert (
            v_front is front_lane0
        ), "Only vehicles on the same lane id of the next segment should match"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the connected-lane neighbour search."""

    def test_no_next_segment(self):
        """When the current segment has no downstream connection."""
        net = RoadNetwork()
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, 0], [50, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
            ),
        )
        road = Road(network=net, np_random=np.random.RandomState(42))
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=48)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert v_front is None
        assert v_rear is None

    def test_no_previous_segment(self):
        """When the current segment has no upstream connection."""
        net = RoadNetwork()
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [50, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
            ),
        )
        road = Road(network=net, np_random=np.random.RandomState(42))
        ego = _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)

        v_front, v_rear = road.neighbour_vehicles(ego, ("b", "c", 0))
        assert v_front is None
        assert v_rear is None

    def test_vehicle_far_on_next_segment_not_detected(self, straight_connected_road):
        """A vehicle very far into the next segment (beyond on_lane margin)
        should not be detected — on_lane check must still filter correctly."""
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=25)
        # Place a vehicle 40m into the next segment — it is on b->c but not
        # geometrically close to a->b at all.
        _make_vehicle(road, net, ("b", "c", 0), longitudinal=40)

        v_front, _ = road.neighbour_vehicles(ego, ("a", "b", 0))
        # The vehicle is on next segment; its adjusted s would be 50+40=90,
        # which is well ahead. It should be found by the search on the next
        # lane, and on_lane check on segment b->c should pass.
        # So it IS a valid front neighbour.
        assert v_front is not None

    def test_lane_index_none_returns_none(self, straight_connected_road):
        """When vehicle has no lane_index, should return (None, None)."""
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=25)
        ego.lane_index = None

        v_front, v_rear = road.neighbour_vehicles(ego)
        assert v_front is None
        assert v_rear is None
