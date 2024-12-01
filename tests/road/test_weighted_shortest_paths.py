import pytest
import numpy as np

import highway_env.envs.weighted_roundabout_env
from highway_env.envs import WeightedRoundaboutEnv, RoundaboutEnv
from highway_env.road.road import WeightedRoadnetwork, PathException
from highway_env.road.lanes.unweighted_lanes import CircularLane, SineLane, StraightLane
from highway_env.road.lanes.lane_utils import LineType


class TestModifiedBellmanFord:
    def test_modified_bellman_ford_with_negative_cycles_plans_route(self):
        # Given
        graph = make_weighted_roundabout(-1)
        src = 'eer'
        target = 'wxr'
        expected_path = ['eer', 'ees', 'ee', 'nx', 'ne', 'wx', 'wxs', 'wxr']

        # When
        actual_path = graph.bellman_ford_negative_cycle(src, target)

        # Then
        assert len(actual_path) > 0
        assert actual_path == expected_path

    def test_modified_bellman_ford_should_raise_on_invalid_route(self):
        # Given
        graph = make_weighted_roundabout(-1)
        src = 'ees'
        target = 'nes'

        # Then
        with pytest.raises(PathException):
            actual_path = graph.bellman_ford_negative_cycle(src, target)

class TestUnmodifiedBellmanFord:
    def test_unmodified_bellman_ford_raises_error_on_negative_cycles(self):
        # Given
        graph = make_weighted_roundabout(-1)
        src = 'eer'
        target = 'wxr'

        # Then
        with pytest.raises(Exception):
            actual_path = graph.bellman_ford(src, target)

    def test_unmodified_bellman_ford_plans_route(self):
        # Given
        graph = make_weighted_roundabout(1)
        src = 'eer'
        target = 'wxr'
        expected_path = ['eer', 'ees', 'ee', 'nx', 'ne', 'wx', 'wxs', 'wxr']

        # When
        actual_path = graph.bellman_ford(src, target)

        # Then
        assert expected_path == actual_path

    def test_unmodified_bellman_ford_should_raise_on_invalid_route(self):
        # Given
        graph = make_weighted_roundabout(1)
        src = 'ees'
        target = 'nes'

        # Then
        with pytest.raises(PathException):
            actual_path = graph.bellman_ford(src, target)

class TestDijkstra:
    def test_dijkstra_raises_error_on_negative_weights(self):
        # Given
        graph = make_weighted_roundabout(-1)
        src = 'eer'
        target = 'wxr'

        # Then
        with pytest.raises(ValueError):
            actual_path = graph.dijkstra(src, target)


    def test_dijkstra_plans_route(self):
        # Given
        graph = make_weighted_roundabout(1)
        src = 'ner'
        target = 'nxr'
        expected_path = ['ner', 'nes', 'ne', 'wx', 'we', 'sx', 'se', 'ex', 'ee', 'nx', 'nxs', 'nxr']

        # When
        actual_path = graph.dijkstra(src, target)

        # Then
        assert len(actual_path) > 0
        assert actual_path == expected_path

    def test_dijkstra_should_raise_on_invalid_route(self):
        # Given
        graph = make_weighted_roundabout(1)
        src = 'eer'
        target = 'nes'

        # Then
        with pytest.raises(PathException):
            actual_path = graph.dijkstra(src, target)


def make_weighted_roundabout(roundabout_lanes_weight: int) -> WeightedRoadnetwork:
    """
    :param roundabout_lanes_weight: negative value, will cause a negative weight cycle
    """
    # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
    center = [0, 0]  # [m]
    radius = 20  # [m]
    alpha = 24  # [deg]

    net = WeightedRoadnetwork()
    radii = [radius, radius + 4]
    n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
    line = [[c, s], [n, c]]
    for lane in [0, 1]:
        # Roundabout itself
        net.add_lane(
            "se",
            "ex",
            CircularLane(
                center,
                radii[lane],
                np.deg2rad(90 - alpha),
                np.deg2rad(alpha),
                clockwise=False,
                line_types=line[lane],
            ),
            roundabout_lanes_weight,
        )
        net.add_lane(
            "ex",
            "ee",
            CircularLane(
                center,
                radii[lane],
                np.deg2rad(alpha),
                np.deg2rad(-alpha),
                clockwise=False,
                line_types=line[lane],
            ),
            roundabout_lanes_weight,
        )
        net.add_lane(
            "ee",
            "nx",
            CircularLane(
                center,
                radii[lane],
                np.deg2rad(-alpha),
                np.deg2rad(-90 + alpha),
                clockwise=False,
                line_types=line[lane],
            ),
            roundabout_lanes_weight,
        )
        net.add_lane(
            "nx",
            "ne",
            CircularLane(
                center,
                radii[lane],
                np.deg2rad(-90 + alpha),
                np.deg2rad(-90 - alpha),
                clockwise=False,
                line_types=line[lane],
            ),
            roundabout_lanes_weight,
        )
        net.add_lane(
            "ne",
            "wx",
            CircularLane(
                center,
                radii[lane],
                np.deg2rad(-90 - alpha),
                np.deg2rad(-180 + alpha),
                clockwise=False,
                line_types=line[lane],
            ),
            roundabout_lanes_weight,
        )
        net.add_lane(
            "wx",
            "we",
            CircularLane(
                center,
                radii[lane],
                np.deg2rad(-180 + alpha),
                np.deg2rad(-180 - alpha),
                clockwise=False,
                line_types=line[lane],
            ),
            1,
        )
        net.add_lane(
            "we",
            "sx",
            CircularLane(
                center,
                radii[lane],
                np.deg2rad(180 - alpha),
                np.deg2rad(90 + alpha),
                clockwise=False,
                line_types=line[lane],
            ),
            roundabout_lanes_weight,
        )
        net.add_lane(
            "sx",
            "se",
            CircularLane(
                center,
                radii[lane],
                np.deg2rad(90 + alpha),
                np.deg2rad(90 - alpha),
                clockwise=False,
                line_types=line[lane],
            ),
            roundabout_lanes_weight,
        )

    # Access lanes: (r)oad/(s)ine
    access = 170  # [m]
    dev = 85  # [m]
    a = 5  # [m]
    delta_st = 0.2 * dev  # [m]

    delta_en = dev - delta_st
    w = 2 * np.pi / dev
    net.add_lane(
        "ser",
        "ses",
        StraightLane(
            [2, access],
            [2, dev / 2],
            line_types=(s, c)),
        1,
    )
    net.add_lane(
        "ses",
        "se",
        SineLane(
            [2 + a, dev / 2],
            [2 + a, dev / 2 - delta_st],
            a,
            w,
            -np.pi / 2,
            line_types=(c, c),
        ),
        1,
    )
    net.add_lane(
        "sx",
        "sxs",
        SineLane(
            [-2 - a, -dev / 2 + delta_en],
            [-2 - a, dev / 2],
            a,
            w,
            -np.pi / 2 + w * delta_en,
            line_types=(c, c),
        ),
        10,
    )
    net.add_lane(
        "sxs", "sxr", weight=1, lane=StraightLane([-2, dev / 2], [-2, access], line_types=(n, c))
    )

    net.add_lane(
        "eer", "ees", weight=1, lane=StraightLane([access, -2], [dev / 2, -2], line_types=(s, c))
    )
    net.add_lane(
        "ees",
        "ee",
        SineLane(
            [dev / 2, -2 - a],
            [dev / 2 - delta_st, -2 - a],
            a,
            w,
            -np.pi / 2,
            line_types=(c, c),
        ),
        1,
    )
    net.add_lane(
        "ex",
        "exs",
        SineLane(
            [-dev / 2 + delta_en, 2 + a],
            [dev / 2, 2 + a],
            a,
            w,
            -np.pi / 2 + w * delta_en,
            line_types=(c, c),
        ),
        10 * roundabout_lanes_weight,
    )
    net.add_lane(
        "exs", "exr", weight=1, lane=StraightLane([dev / 2, 2], [access, 2], line_types=(n, c))
    )

    net.add_lane(
        "ner", "nes", weight=1, lane=StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c))
    )
    net.add_lane(
        "nes",
        "ne",
        SineLane(
            [-2 - a, -dev / 2],
            [-2 - a, -dev / 2 + delta_st],
            a,
            w,
            -np.pi / 2,
            line_types=(c, c),
        ),
        1,
    )
    net.add_lane(
        "nx",
        "nxs",
        SineLane(
            [2 + a, dev / 2 - delta_en],
            [2 + a, -dev / 2],
            a,
            w,
            -np.pi / 2 + w * delta_en,
            line_types=(c, c),
        ),
        10 * roundabout_lanes_weight,
    )
    net.add_lane(
        "nxs", "nxr", weight=1, lane=StraightLane([2, -dev / 2], [2, -access], line_types=(n, c))
    )

    net.add_lane(
        "wer", "wes", weight=1, lane=StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c))
    )
    net.add_lane(
        "wes",
        "we",
        SineLane(
            [-dev / 2, 2 + a],
            [-dev / 2 + delta_st, 2 + a],
            a,
            w,
            -np.pi / 2,
            line_types=(c, c),
        ),
        1,
    )
    net.add_lane(
        "wx",
        "wxs",
        SineLane(
            [dev / 2 - delta_en, -2 - a],
            [-dev / 2, -2 - a],
            a,
            w,
            -np.pi / 2 + w * delta_en,
            line_types=(c, c),
        ),
        10 * roundabout_lanes_weight,
    )
    net.add_lane(
        "wxs", "wxr", weight=1, lane=StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c))
    )

    return net