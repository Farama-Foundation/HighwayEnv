from __future__ import annotations

import numpy as np
from typing_extensions import override

from highway_env import utils
from highway_env.envs import RoundaboutEnv
from highway_env.road.lanes.unweighted_lanes import CircularLane, SineLane, StraightLane
from highway_env.road.lanes.lane_utils import LineType, LaneType
from highway_env.road.road import Road, WeightedRoadnetwork

class WeightedRoundaboutEnv(RoundaboutEnv):
    @override
    def _make_road(self) -> None:
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
                -1,
                LaneType.ROUNDABOUT,
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
                -1,
                LaneType.ROUNDABOUT,
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
                -1,
                LaneType.ROUNDABOUT,
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
                -1,
                LaneType.ROUNDABOUT,
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
                -1,
                LaneType.ROUNDABOUT,
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
                LaneType.ROUNDABOUT,
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
                -1,
                LaneType.ROUNDABOUT,
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
                -1,
                LaneType.ROUNDABOUT,
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
            LaneType.ROUNDABOUT,
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
            LaneType.ROUNDABOUT,
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
            LaneType.ROUNDABOUT,
        )
        net.add_lane(
            "sxs", "sxr",
            weight=1,
            lane=StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)),
            lane_type=LaneType.ROUNDABOUT
        )

        net.add_lane(
            "eer",
            "ees",
            weight=1,
            lane=StraightLane([access, -2], [dev / 2, -2], line_types=(s, c)),
            lane_type=LaneType.ROUNDABOUT,
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
            LaneType.ROUNDABOUT,
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
            -10,
            LaneType.ROUNDABOUT,
        )
        net.add_lane(
            "exs",
            "exr",
            weight=1,
            lane=StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)),
            lane_type=LaneType.ROUNDABOUT,
        )

        net.add_lane(
            "ner",
            "nes",
            weight=1,
            lane=StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)),
            lane_type=LaneType.ROUNDABOUT,
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
            LaneType.ROUNDABOUT,
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
            -10,
            LaneType.ROUNDABOUT,
        )
        net.add_lane(
            "nxs",
            "nxr",
            weight=1,
            lane=StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)),
            lane_type=LaneType.ROUNDABOUT,
        )

        net.add_lane(
            "wer",
            "wes",
            weight=1,
            lane=StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)),
            lane_type=LaneType.ROUNDABOUT,
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
            LaneType.ROUNDABOUT,
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
            -10,
            LaneType.ROUNDABOUT,
        )
        net.add_lane(
            "wxs",
                "wxr",
            weight=1,
            lane=StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)),
            lane_type=LaneType.ROUNDABOUT,
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road