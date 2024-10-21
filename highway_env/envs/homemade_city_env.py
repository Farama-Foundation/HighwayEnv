from __future__ import annotations
import math

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.utils import Vector

from highway_env.road.lane import AbstractLane, CircularLane, LineType, SineLane, StraightLane

from highway_env.road.regulation import RegulatedRoad


class HomemadeCity(AbstractEnv):
    """
    A testing driving environment.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "LidarObservation",
                    "cells": 360,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": [-10, 0, 10, 20, 30, 40, 50, 60] #np.linspace(0, 30, 10)
                },
                "simulation_frequency": 15,
                "lanes_count": 2,
                "vehicles_count": 0,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 60,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
                "screen_height": 600,
                "screen_width": 1200,
            }
        )
        return config    

    def _reset(self) -> None:
            self._make_road()
            self._make_vehicles()
    
    # Perpendicular Bisector
    def _find_circle_center(self, start: Vector, end: Vector) -> Vector:
        x1, y1 = start
        x2, y2 = end
        
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        if x1 == x2:
            return (mid_x, None)
        elif y1 == y2:
            return (None, mid_y)

        slope = (y2 - y1) / (x2 - x1)
        perp_slope = -1 / slope

        radius = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2

        dx = math.sqrt(radius**2 / (1 + perp_slope**2))
        dy = perp_slope * dx

        center1 = (mid_x + dx, mid_y + dy)
        center2 = (mid_x - dx, mid_y - dy)

        return center1, center2
    
    def _cross_product(self, v1: Vector, v2: Vector):
        """Compute the 2D cross product of two vectors v1 and v2."""
        return v1[0] * v2[1] - v1[1] * v2[0]
    
    def _select_center(
        self,
        start: Vector,
        end: Vector,
        center1: Vector,
        center2: Vector,
        turn_direction: bool
        ) -> Vector:
        
        start_x, start_y = start
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        vector_start_to_mid = (mid_x - start_x, mid_y - start_y)
        vector_mid_to_center1 = (center1[0] - mid_x, center1[1] - mid_y)
        
        cross = self._cross_product(vector_start_to_mid, vector_mid_to_center1)
        
        # Check direction based on the sign of the cross product
        if turn_direction:
            # For turn_direction, cross product should be negative
            if cross < 0:
                return center1
            else:
                return center2
        else:
            # For counterturn_direction, cross product should be positive
            if cross > 0:
                return center1
            else:
                return center2

    def _get_center(self, start: Vector, end: Vector, turn_direction: bool) -> Vector:
        # turn_direction:: False->left_turn ; True->right_turn
        center1, center2 = self._find_circle_center(start, end)
        
        return self._select_center(start, end, center1, center2, not(turn_direction))
    
    def _get_radius(self, start: Vector, center: Vector) -> float:
        return math.sqrt((start[0] - center[0])**2 + (start[1] - center[1])**2)
            
    def _build_circlelane(
        self,
        start: Vector,
        end: Vector,
        turn_direction: bool,
        start_phase: int,
        end_phase: int,
        priority: int,
        line_types: list[LineType] = None,
        lane_width: float = AbstractLane.DEFAULT_WIDTH,
        forbidden: bool = False,
        speed_limit: float = 20
        ) -> CircularLane:
        
        center = self._get_center(start, end, turn_direction)
        
        return CircularLane(
            center,
            self._get_radius(start, center),
            np.deg2rad(start_phase),
            np.deg2rad(end_phase),
            turn_direction,
            lane_width,
            line_types,
            forbidden,
            speed_limit,
            priority,
        )
            
    def _make_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""

        net = RoadNetwork()
        line_none, line_continuous, line_striped = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        turn_radius = 4 # [m]
        roundabout_radius = 20 # [m]
        left_turn = False
        right_turn = True
        lane_width = AbstractLane.DEFAULT_WIDTH
        
        line_type = [[line_continuous, line_striped], [line_none, line_continuous]]
        line_type_merge = [[line_continuous, line_striped], [line_none, line_striped]]
        lane_placement = [0, lane_width]
        
        
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [roundabout_radius, roundabout_radius + lane_width]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
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
            )

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2 * dev  # [m]
        
        delta_en = dev - delta_st
        w = 2 * np.pi / dev
        # South Enter straigth lane
        net.add_lane(
            "ser",
            "ses",
            StraightLane(
                [2, 80],
                [2, dev / 2],
                line_types=(s, c),
                priority=3
            )
        )
        # South Enter sine lane
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
        )
        # South Exit sine lane
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
        )
        # South Exit straight lane
        net.add_lane(
            "sxs",
            "sxr",
            StraightLane(
                [-2, dev / 2],
                [-2, 80],
                line_types=(n, c),
                priority=3
            )
        )

        # East Enter straight lane
        net.add_lane(
            "I-4:w-in",
            "ees",
            StraightLane(
                [120, -2],
                [dev / 2, -2],
                line_types=(s, c),
                priority=3
            )
        )
        # East Enter sine lane
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
        )
        # East Exit sine lane
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
        )
        # East Exit straight lane
        net.add_lane(
            "exs",
            "I-4:w-out",
            StraightLane(
                [dev / 2, 2],
                [120, 2],
                line_types=(n, c),
                priority=3
            )
        )

        # North Enter straight lane
        net.add_lane(
            "I-1:s-in",
            "nes",
            StraightLane(
                [-2, -80],
                [-2, -dev / 2],
                line_types=(s, c),
                priority=3
            )
        )
        # Nort Enter sine lane
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
        )
        # Nort Exit sine lane
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
        )
        # North Exit straight lane
        net.add_lane(
            "nxs",
            "I-1:s-out",
            StraightLane(
                [2, -dev / 2],
                [2, -80],
                line_types=(n, c),
                priority=3
            )
        )

        # West Enter straight lane
        net.add_lane(
            "I-12:e-in",
            "wes",
            StraightLane(
                [-116, 2],
                [-dev / 2, 2],
                line_types=(s, c),
                priority=3
            )
        )
        # West Enter sine lane
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
        )
        # West Exit sine lane
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
        )
        # West Exit straight lane
        net.add_lane(
            "wxs",
            "I-12:e-out",
            StraightLane(
                [-dev / 2, -2],
                [-116, -2],
                line_types=(n, c),
                priority=3
            )
        )   


        """Intersection 12"""
        # I-12 - east to west
        net.add_lane(
            "I-12:e-out",
            "I-12:w-in",
            StraightLane(
                [-116, -2],
                [-128, -2],
                lane_width,
                (n,c),
                priority=3
            )
        )
        # I-12 - west to east
        net.add_lane(
            "I-12:w-out",
            "I-12:e-in",
            StraightLane(
                [-128, 2],
                [-116, 2],
                lane_width,
                (s,n),
                priority=3
            )
        )

        # I-12 - south to east
        net.add_lane(
            "I-12:s-out",
            "I-12:e-in",
            CircularLane(
                [-116, 6],
                turn_radius,
                np.deg2rad(-180),
                np.deg2rad(-90),
                right_turn,
                lane_width,
                (n, c),
                priority=0
            )
        )
        # I-12 - south to west
        net.add_lane(
            "I-12:s-out",
            "I-12:w-in",
            CircularLane(
                [-128, 6],
                turn_radius + lane_width,
                np.deg2rad(0),
                np.deg2rad(-90),
                left_turn,
                lane_width,
                (n, n),
                priority=0
            )
        )
        
        # I-12 - east to south
        net.add_lane(
            "I-12:e-out",
            "I-12:s-in",
            CircularLane(
                [-116, 6],
                turn_radius + lane_width,
                np.deg2rad(-90),
                np.deg2rad(-180),
                left_turn,
                lane_width,
                (n, n),
                priority=0
            )
        )
        # I-12 - west to south
        net.add_lane(
            "I-12:w-out",
            "I-12:s-in",
            CircularLane(
                [-128, 6],
                turn_radius,
                np.deg2rad(-90),
                np.deg2rad(0),
                right_turn,
                lane_width,
                (n,c),
                priority=1
            )
        )


        """Turn 1"""
        # T-1 - east to north
        net.add_lane(
            "I-12:w-in",
            "R-19:s-in",
            CircularLane(
                [-128, -6],
                turn_radius,
                np.deg2rad(90),
                np.deg2rad(180),
                right_turn,
                line_types=(n, c)                
            )
        )
        # T-1 - north to east
        net.add_lane(
            "R-19:s-out",
            "I-12:w-out",
            CircularLane(
                [-128,-6],
                turn_radius + lane_width,
                np.deg2rad(90),
                np.deg2rad(180),
                right_turn,
                line_types=(c, s)
            )
        )
        
        
        """Road 19"""
        # R-19 - south to north
        net.add_lane(
            "R-19:s-in",
            "I-13:s-out",
            StraightLane(
                [-132, -6],
                [-132, -80],
                lane_width,
                (n, c),
                priority=3
            )
        )
        # R-19 - north to south
        net.add_lane(
            "I-13:s-in",
            "R-19:s-out",
            StraightLane(
                [-136, -80],
                [-136, -6],
                lane_width,
                (s, c),
                priority=3
            )
        )
        
        
        """Intersection 13"""
        # I-13 - east to west
        net.add_lane(
            "I-13:e-out",
            "I-13:w-in",
            StraightLane(
                [-128, -88],
                [-140, -88],
                lane_width,
                (n, c),
                priority=3
            )
        )
        # I-13 - west to east
        net.add_lane(
            "I-13:w-out",
            "I-13:e-in",
            StraightLane(
                [-140, -84],
                [-128, -84],
                lane_width,
                (s, n),
                priority=3
            )
        )
        
        # I-13 - east to south
        net.add_lane(
            "I-13:e-out",
            "I-13:s-in",
            CircularLane(
                [-128, -80],
                turn_radius + lane_width,
                np.deg2rad(-90),
                np.deg2rad(-180),
                left_turn,
                lane_width,
                (n, n),
                priority=0
            )
        )
        # I-13 - west to south
        net.add_lane(
            "I-13:w-out",
            "I-13:s-in",
            CircularLane(
                [-140, -80],
                turn_radius,
                np.deg2rad(-90),
                np.deg2rad(0),
                right_turn,
                lane_width,
                (n, c),
                priority=1
            )
        )
        # I-13 - south to west
        net.add_lane(
            "I-13:s-out",
            "I-13:w-in",
            CircularLane(
                [-140, -80],
                turn_radius + lane_width,
                np.deg2rad(0),
                np.deg2rad(-90),
                left_turn,
                lane_width,
                (n, n),
                priority=0
            )
        )
        # I-13 - south to east
        net.add_lane(
            "I-13:s-out",
            "I-13:e-in",
            CircularLane(
                [-128, -80],
                turn_radius,
                np.deg2rad(-180),
                np.deg2rad(-90),
                right_turn,
                lane_width,
                (n, c),
                priority=0
            )
        )
        
        
        """Exit 1"""
        # E-1 - east to west
        net.add_lane(
            "E-1:out",
            "I-13:w-out",
            StraightLane(
                [-200, -84],
                [-140, -84],
                lane_width,
                (s, c),
                priority=3
            )
        )
        # E-1 - west to east
        net.add_lane(
            "I-13:w-in",
            "E-1:in",
            StraightLane(
                [-140, -88],
                [-200, -88],
                lane_width,
                (n, c),
                priority=3
            )
        )
        # E-1 - turn car around
        net.add_lane(
            "E-1:in",
            "E-1:out",
            CircularLane(
                [-200, -84],
                4,
                np.deg2rad(-90),
                np.deg2rad(-270),
                left_turn,
                lane_width,
                (n, c)
            )
        )
        
        
        """Road 1"""
        # R-1 - west to east
        net.add_lane(
            "I-13:e-in",
            "I-1:w-out",
            StraightLane(
                [-128, -84],
                [-6, -84],
                lane_width,
                (n,c),
                priority=3
            )
        )
        # R-1 - east to west
        net.add_lane(
            "I-1:w-in",
            "I-13:e-out",
            StraightLane(
                [-6, -88],
                [-128, -88],
                lane_width,
                (s, c),
                priority=3
            )
        )
        
        
        """Intersection 1"""
        # I-1 - west to east
        net.add_lane(
            "I-1:w-out",
            "I-1:e-in",
            StraightLane(
                [-6, -84],
                [7, -84],
                lane_width,
                (s,n),
                priority=3
            )
        )
        # I-1 - west to north
        net.add_lane(
            "I-1:w-out",
            "I-1:n-in",
            self._build_circlelane(
                [-6, -84],
                [2, -92],
                left_turn,
                90,
                0,
                2,
                (n,n)
            )
        )
        # I-1 - west to south
        net.add_lane(
            "I-1:w-out",
            "I-1:s-in",
            self._build_circlelane(
                [-6, -84],
                [-2, -80],
                right_turn,
                -90,
                0,
                3,
                (n,c)
            )
        )
        
        # I-1 - east to west
        net.add_lane(
            "I-1:e-out",
            "I-1:w-in",
            StraightLane(
                [6, -88],
                [-6, -88],
                lane_width,
                (n,n),
                priority=3
            )
        )
        # I-1 - east to north
        net.add_lane(
            "I-1:e-out",
            "I-1:n-in",
            self._build_circlelane(
                [6, -88],
                [2, -92],
                right_turn,
                90,
                180,
                3,
                (n,c)
            )
        )
        # I-1 - east to south
        net.add_lane(
            "I-1:e-out",
            "I-1:s-in",
            self._build_circlelane(
                [6, -88],
                [-2, -80],
                left_turn,
                -90,
                -180,
                2,
                (n,n)
            )
        )
        
        # I-1 - south to north
        net.add_lane(
            "I-1:s-out",
            "I-1:n-in",
            StraightLane(
                [2, -80],
                [2, -92],
                lane_width,
                (n,n),
                priority=0
            )
        )
        # I-1 - south to west
        net.add_lane(
            "I-1:s-out",
            "I-1:w-in",
            self._build_circlelane(
                [2, -80],
                [-6, -88],
                left_turn,
                0,
                -90,
                1,
                (n,n)
            )
        )
        # I-1 - south to east
        net.add_lane(
            "I-1:s-out",
            "I-1:e-in",
            self._build_circlelane(
                [2, -80],
                [6, -84],
                right_turn,
                -180,
                -90,
                1,
                (n,c)
            )
        )
        
        # I-1 - north to south
        net.add_lane(
            "I-1:n-out",
            "I-1:s-in",
            StraightLane(
                [-2, -92],
                [-2, -80],
                lane_width,
                (n,n),
                priority=0
            )
        )
        # I-1 - north to west
        net.add_lane(
            "I-1:n-out",
            "I-1:w-in",
            self._build_circlelane(
                [-2, -92],
                [-6, -88],
                right_turn,
                0,
                90,
                1,
                (n,c)
            )
        )
        # I-1 - north to east
        net.add_lane(
            "I-1:n-out",
            "I-1:e-in",
            self._build_circlelane(
                [-2, -92],
                [6, -84],
                left_turn,
                180,
                90,
                1,
                (n,n)
            )
        )
        
        
        """Exit 2"""
        # E-2 - south to north
        net.add_lane(
            "I-1:n-in",
            "E-2:out",
            StraightLane(
                [2, -92],
                [2, -150],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # E-2 - south to north
        net.add_lane(
            "E-2:in",
            "I-1:n-out",
            StraightLane(
                [-2, -150],
                [-2, -92],
                lane_width,
                (n,c),
                priority=3
            )
        )
        # E-2 - turn car around
        net.add_lane(
            "E-2:out",
            "E-2:in",
            CircularLane(
                [1, -150],
                4,
                np.deg2rad(0),
                np.deg2rad(-180),
                left_turn,
                lane_width,
                (n, c)
            )
        )
        
        
        """Road 2"""
        # R-2 - west to east
        net.add_lane(
            "I-1:e-in",
            "I-2:w-out",
            StraightLane(
                [6, -84],
                [60, -84],
                lane_width,
                (n,c),
                priority=3
            )
        )
        # R-2 - east to west
        net.add_lane(
            "I-2:w-in",
            "I-1:e-out",
            StraightLane(
                [60, -88],
                [6, -88],
                lane_width,
                (s, c),
                priority=3
            )
        )
        
        
        """Intersection 2"""
        # I-2 - west to nort
        net.add_lane(
            "I-2:w-out",
            "I-2:n-in",
            self._build_circlelane(
                [60, -84],
                [68, -92],
                left_turn,
                90,
                0,
                0,
                (n,n)
            )
        )
        # I-2 - west to south
        net.add_lane(
            "I-2:w-out",
            "I-2:s-in",
            self._build_circlelane(
                [60, -84],
                [64, -80],
                right_turn,
                -90,
                0,
                0,
                (n,c)
            )
        )
        
        # I-2 - north to south
        net.add_lane(
            "I-2:n-out",
            "I-2:s-in",
            StraightLane(
                [64, -92],
                [64, -80],
                lane_width,
                (s,n),
                priority=3
            )
        )
        # I-2 - north to west
        net.add_lane(
            "I-2:n-out",
            "I-2:w-in",
            self._build_circlelane(
                [64, -92],
                [60, -88],
                right_turn,
                0,
                90,
                3,
                (n,c)
            )
        )
        
        # I-2 - south to north
        # net.add_lane(
        #     "I-2:s-out",
        #     "I-2:n-in",
        #     StraightLane(
        #         [68, -80],
        #         [68, -92],
        #         lane_width,
        #         (n,c),
        #         priority=3
        #     )
        # )
        # I-2 - south to west
        # net.add_lane(
        #     "I-2:s-out",
        #     "I-2:w-in",
        #     CircularLane(
        #         [60, -80],
        #         8,
        #         0,
        #         90,
        #         left_turn,
        #         lane_width,
        #         (c,c),
        #         priority=2
        #     )
        #     # self._build_circlelane(
        #     #     [68, -80],
        #     #     [60, -88],
        #     #     left_turn,
        #     #     0,
        #     #     -90,
        #     #     2,
        #     #     (c,c)
        #     # )
        # )
        
        
        """Exit 3"""
        # E-3 - south to north
        net.add_lane(
            "I-2:n-in",
            "E-3:out",
            StraightLane(
                [68, -92],
                [68, -150],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # E-3 - south to north
        net.add_lane(
            "E-3:in",
            "I-2:n-out",
            StraightLane(
                [64, -150],
                [64, -92],
                lane_width,
                (n,c),
                priority=3
            )
        )
        # E-3 - turn car around
        net.add_lane(
            "E-3:out",
            "E-3:in",
            CircularLane(
                [63, -150],
                4,
                np.deg2rad(0),
                np.deg2rad(-180),
                left_turn,
                lane_width,
                (n, c)
            )
        )
        
        
        """Road 3"""
        # R-3 - north to south
        net.add_lane(
            "I-2:s-in",
            "T-2:n-out",
            StraightLane(
                [64, -80],
                [64, -45],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # R-3 - south to north
        net.add_lane(
            "T-2:n-in",
            "I-1:s-out",
            StraightLane(
                [68, -45],
                [68, -80],
                lane_width,
                (n,c),
                priority=3
            )
        )
        
        
        """Turn 2"""
        # T-2 - north to east
        net.add_lane(
            "T-2:n-out",
            "T-2:e-in",
            self._build_circlelane(
                [64, -45],
                [72, -37],
                left_turn,
                180,
                90,
                0,
                (s,c)
            )
        )
        # T-2 - east to north
        net.add_lane(
            "T-2:e-out",
            "T-2:n-in",
            self._build_circlelane(
                [72, -41],
                [68, -45],
                right_turn,
                90,
                180,
                0,
                (n,c)
            )
        )
        
        
        """Road 4"""
        # R-4 - west to east
        net.add_lane(
            "T-2:e-in",
            "I-3:w-out",
            StraightLane(
                [72, -37],
                [120, -37],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # R-4 - east to west
        net.add_lane(
            "I-3:w-in",
            "T-2:e-out",
            StraightLane(
                [120, -41],
                [72, -41],
                lane_width,
                (n,c),
                priority=3
            )
        )
        
        
        """Intersection 3"""
        # I-3 - west to east
        net.add_lane(
            "I-3:w-out",
            "I-3:e-in",
            StraightLane(
                [120, -37],
                [132, -37],
                lane_width,
                (s,n),
                priority=3
            )
        )
        # I-3 - west to south
        net.add_lane(
            "I-3:w-out",
            "I-3:s-in",
            self._build_circlelane(
                [120, -37],
                [124, -33],
                right_turn,
                -90,
                0,
                3,
                (n,c)
            )
        )
        
        # I-3 - south to west
        net.add_lane(
            "I-3:s-out",
            "I-3:w-in",
            self._build_circlelane(
                [128, -33],
                [120, -41],
                left_turn,
                0,
                -90,
                0,
                (n,n)
            )
        )
        # I-3 - south to east
        net.add_lane(
            "I-3:s-out",
            "I-3:e-in",
            self._build_circlelane(
                [128, -33],
                [132, -37],
                right_turn,
                -180,
                -90,
                0,
                (n,c)
            )
        )

        # I-3 - east to west
        net.add_lane(
            "I-3:e-out",
            "I-3:w-in",
            StraightLane(
                [132, -41],
                [120, -41],
                lane_width,
                (n,c),
                priority=3
            )
        ) 
        # I-3 - east to south
        net.add_lane(
            "I-3:e-out",
            "I-3:s-in",
            self._build_circlelane(
                [132, -41],
                [124, -33],
                left_turn,
                -90,
                -180,
                2,
                (n,n)
            )
        )
               
        
        """Exit 4"""        
        # E-4 - west to east
        net.add_lane(
            "I-3:e-in",
            "E-4:out",
            StraightLane(
                [132, -37],
                [170, -37],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # E-4 - east to west
        net.add_lane(
            "E-4:in",
            "I-3:e-out",
            StraightLane(
                [170, -41],
                [132, -41],
                lane_width,
                (n,c),
                priority=3
            )
        )
        # E-4 - turn car around
        net.add_lane(
            "E-4:out",
            "E-4:in",
            CircularLane(
                [169, -41],
                4,
                np.deg2rad(90),
                np.deg2rad(-90),
                left_turn,
                lane_width,
                (n, c)
            )
        )
        
        
        """Road 5"""
        # R-5 - north to south
        net.add_lane(
            "I-3:s-in",
            "I-4:n-out",
            StraightLane(
                [124, -33],
                [124, -6],
                lane_width,
                (s,c)
            )
        )
        # R-5 - south to north
        net.add_lane(
            "I-4:n-in",
            "I-3:s-out",
            StraightLane(
                [128, -6],
                [128, -33],
                lane_width,
                (n,c)
            )
        )
        
        """Intersection 4"""
        # I-4 - north to south
        net.add_lane(
            "I-4:n-out",
            "I-4:s-in",
            StraightLane(
                [124, -6],
                [124, 6],
                lane_width,
                (s,n),
                priority=3
            )
        )
        # I-4 - north to west
        net.add_lane(
            "I-4:n-out",
            "I-4:s-in",
            self._build_circlelane(
                [124, -6],
                [120, -2],
                right_turn,
                0,
                90,
                3,
                (n,c)
            )
        )

        # I-4 - west to north
        net.add_lane(
            "I-4:w-out",
            "I-4:n-in",
            self._build_circlelane(
                [120, 2],
                [128, -6],
                left_turn,
                90,
                0,
                0,
                (n,n)
            )
        )
        # I-4 - west to south
        net.add_lane(
            "I-4:w-out",
            "I-4:s-in",
            self._build_circlelane(
                [120, 2],
                [124, 6],
                right_turn,
                -90,
                0,
                0,
                (n,c)
            )
        )
        
        # I-4 - south to north
        net.add_lane(
            "I-4:s-out",
            "I-4:n-in",
            StraightLane(
                [128, 6],
                [128, -6],
                lane_width,
                (n,c),
                priority=3
            )
        )
        # I-4 - south to west
        net.add_lane(
            "I-4:s-out",
            "I-4:w-in",
            self._build_circlelane(
                [128, 6],
                [120, -2],
                left_turn,
                0,
                -90,
                2,
                (n,n)
            )
        )
        
        
        """Road 6"""
        # R-6 - north to south
        net.add_lane(
            "I-4:s-in",
            "I-5:n-out",
            StraightLane(
                [124, 6],
                [124, 16],
                lane_width,
                (s,c)
            )
        )
        # R-6 - south to north
        net.add_lane(
            "I-5:n-in",
            "I-4:s-out",
            StraightLane(
                [128, 16],
                [128, 6],
                lane_width,
                (n,c)
            )
        )
        
        
        """Intersection 5"""
        # I-5 - north to south
        net.add_lane(
            "I-5:n-out",
            "I-5:s-in",
            StraightLane(
                [124, 16],
                [124, 28],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # I-5 - north to east
        net.add_lane(
            "I-5:n-out",
            "I-5:e-in",
            self._build_circlelane(
                [124, 16],
                [132, 24],
                left_turn,
                180,
                90,
                2,
                (n,n)
            )
        )

        # I-5 - east to north
        net.add_lane(
            "I-5:e-out",
            "I-5:n-in",
            self._build_circlelane(
                [132, 20],
                [128, 16],
                right_turn,
                90,
                180,
                0,
                (n,c)
            )
        )
        # I-5 - east to south
        net.add_lane(
            "I-5:e-out",
            "I-5:s-in",
            self._build_circlelane(
                [132, 20],
                [124, 28],
                left_turn,
                -90,
                -180,
                0,
                (n,n)
            )
        )

        # I-5 - south to north
        net.add_lane(
            "I-5:s-out",
            "I-5:n-in",
            StraightLane(
                [128, 28],
                [128, 16],
                lane_width,
                (n,n),
                priority=3
            )
        )
        # I-5 - south to east
        net.add_lane(
            "I-5:s-out",
            "I-5:e-in",
            self._build_circlelane(
                [128, 28],
                [132, 24],
                right_turn,
                -180,
                -90,
                3,
                (n,c)
            )
        )

        
        """Exit 5"""
        # E-5 - west to east
        net.add_lane(
            "I-5:e-in",
            "E-5:in",
            StraightLane(
                [132, 24],
                [170, 24],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # E-5 - east to west
        net.add_lane(
            "E-5:out",
            "I-5:e-out",
            StraightLane(
                [170, 20],
                [132, 20],
                lane_width,
                (n,c),
                priority=3
            )
        )
        # E-5 - turn car around
        net.add_lane(
            "E-5:in",
            "E-5:out",
            CircularLane(
                [169, 20],
                4,
                np.deg2rad(90),
                np.deg2rad(-90),
                left_turn,
                lane_width,
                (n, c)
            )
        )
        
        
        """Road 7"""
        # R-7 - north to south
        net.add_lane(
            "I-5:s-in",
            "I-6:n-out",
            StraightLane(
                [124, 28],
                [124, 38],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # R-7 - south to north
        net.add_lane(
            "I-6:n-in",
            "I-5:s-out",
            StraightLane(
                [128, 38],
                [128, 28],
                lane_width,
                (n,c),
                priority=3
            )
        )
        
        
        """Intersection 6"""
        # I-6 - north to south
        net.add_lane(
            "I-6:n-out",
            "I-6:s-in",
            StraightLane(
                [124, 38],
                [124, 50],
                lane_width,
                (s,n),
                priority=3
            )
        )
        # I-6 - north to west
        net.add_lane(
            "I-6:n-out",
            "I-6:w-in",
            self._build_circlelane(
                [124, 38],
                [120, 42],
                right_turn,
                0,
                90,
                3,
                (n,c)
            )
        )

        # I-6 - west to north
        net.add_lane(
            "I-6:w-out",
            "I-6:n-in",
            self._build_circlelane(
                [120, 46],
                [128, 38],
                left_turn,
                90,
                0,
                0,
                (n,n)
            )
        )
        # I-6 - west to south
        net.add_lane(
            "I-6:w-out",
            "I-6:s-in",
            self._build_circlelane(
                [120, 46],
                [124, 50],
                right_turn,
                -90,
                0,
                0,
                (n,c)
            )
        )

        # I-6 - south to north
        net.add_lane(
            "I-6:s-out",
            "I-6:n-in",
            StraightLane(
                [128, 50],
                [128, 38],
                lane_width,
                (n,c),
                priority=3
            )
        )
        # I-6 - south to west
        net.add_lane(
            "I-6:s-out",
            "I-6:w-in",
            self._build_circlelane(
                [128, 50],
                [120, 42],
                left_turn,
                0,
                -90,
                2,
                (n,n)
            )
        )

        
        """Road 11"""
        # R-11 - east to west
        net.add_lane(
            "I-6:w-in",
            "T-3:e-out",
            StraightLane(
                [120, 42],
                [60, 42],
                lane_width,
                (s,c)
            )
        )
        # R-11 - west to east
        net.add_lane(
            "T-3:e-in",
            "I-6:w-out",
            StraightLane(
                [60, 46],
                [120, 46],
                lane_width,
                (n,c)
            )
        )
        
        
        """Road 8"""
        # R-8 - north to south
        net.add_lane(
            "I-6:s-in",
            "I-7:n-out",
            StraightLane(
                [124, 50],
                [124, 100],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # R-8 - south to north
        net.add_lane(
            "I-7:n-in",
            "I-6:s-out",
            StraightLane(
                [128, 100],
                [128, 50],
                lane_width,
                (n,c),
                priority=3
            )
        )
        
        
        """Intersection 7"""
        # I-7 - north to west
        net.add_lane(
            "I-7:n-out",
            "I-7:w-in",
            self._build_circlelane(
                [124, 100],
                [120, 104],
                right_turn,
                0,
                90,
                0,
                (n,c)
            )
        )
        # I-7 - north to east
        net.add_lane(
            "I-7:n-out",
            "I-7:e-in",
            self._build_circlelane(
                [124, 100],
                [132, 108],
                left_turn,
                180,
                90,
                0,
                (n,n)
            )
        )
        
        # I-7 - west to east
        net.add_lane(
            "I-7:w-out",
            "I-7:e-in",
            StraightLane(
                [120, 108],
                [132, 108],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # I-7 - west to north
        net.add_lane(
            "I-7:w-out",
            "I-7:n-in",
            self._build_circlelane(
                [120, 108],
                [128, 100],
                left_turn,
                90,
                0,
                2,
                (n,n)
            )
        )
        
        # I-7 - east to west
        net.add_lane(
            "I-7:e-out",
            "I-7:w-in",
            StraightLane(
                [132, 104],
                [120, 104],
                lane_width,
                (n,n),
                priority=3
            )
        )
        # I-7 - east to north
        net.add_lane(
            "I-7:w-out",
            "I-7:n-in",
            self._build_circlelane(
                [132, 104],
                [128, 100],
                right_turn,
                90,
                180,
                3,
                (n,c)
            )
        )
        
        
        """Exit 6"""
        # E-6 - west to east
        net.add_lane(
            "I-7:e-in",
            "E-6:out",
            StraightLane(
                [132, 108],
                [170, 108],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # E-6 - east to west
        net.add_lane(
            "E-6:in",
            "I-7:e-out",
            StraightLane(
                [170, 104],
                [132, 104],
                lane_width,
                (n,c),
                priority=3
            )
        )
        # E-6 - turn car around
        net.add_lane(
            "E-6:out",
            "E-6:in",
            CircularLane(
                [169, 104],
                4,
                np.deg2rad(90),
                np.deg2rad(-90),
                left_turn,
                lane_width,
                (n, c)
            )
        )
        
        
        """Road 9"""
        # R-9 - west to east
        net.add_lane(
            "I-7:w-in",
            "I-9:e-out",
            StraightLane(
                [120, 104],
                [60, 104],
                lane_width,
                (s,c)
            )
        )
        # R-9 - east to west
        net.add_lane(
            "I-9:e-in",
            "I-7:w-out",
            StraightLane(
                [60, 108],
                [120, 108],
                lane_width,
                (n,c)
            )
        )
        
        
        """Intersection 9"""
        # I-9 - east to north
        net.add_lane(
            "I-9:e-out",
            "I-9:n-in",
            self._build_circlelane(
                [60, 104],
                [56, 100],
                right_turn,
                90,
                180,
                2,
                (n,c)
            )
        )
        # I-9 - east to south
        net.add_lane(
            "I-9:e-out",
            "I-9:s-in",
            self._build_circlelane(
                [60, 104],
                [52, 112],
                left_turn,
                -90,
                -180,
                0,
                (n,n)
            )
        )
        
        # I-9 - north to south
        net.add_lane(
            "I-9:n-out",
            "I-9:s-in",
            StraightLane(
                [52, 100],
                [52, 112],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # I-9 - north to east
        net.add_lane(
            "I-9:n-out",
            "I-9:e-in",
            self._build_circlelane(
                [52, 100],
                [60, 108],
                left_turn,
                180,
                90,
                0,
                (n,n)
            )
        )
        
        # I-9 - south to north
        net.add_lane(
            "I-9:s-out",
            "I-9:n-in",
            StraightLane(
                [56, 112],
                [56, 100],
                lane_width,
                (n,n),
                priority=3
            )
        )
        # I-9 - south to east
        net.add_lane(
            "I-9:s-out",
            "I-9:e-in",
            self._build_circlelane(
                [56, 112],
                [60, 108],
                right_turn,
                -180,
                -90,
                2,
                (n,c)
            )
        )
        
        
        """Exit 7"""
        # E-7 - north to south
        net.add_lane(
            "I-9:s-in",
            "E-7:out",
            StraightLane(
                [52, 112],
                [52, 140],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # E-7 - south to north
        net.add_lane(
            "E-7:in",
            "I-9:s-out",
            StraightLane(
                [56, 140],
                [56, 112],
                lane_width,
                (n,c),
                priority=3
            )
        )
        # E-7 - turn car around
        net.add_lane(
            "E-7:out",
            "E-7:in",
            CircularLane(
                [56, 140],
                4,
                np.deg2rad(180),
                np.deg2rad(0),
                left_turn,
                lane_width,
                (n, c)
            )
        )
        
        
        """Road 10"""
        # R-10 - north to south
        net.add_lane(
            "I-8:s-in",
            "I-9:n-out",
            StraightLane(
                [52, 92],
                [52, 100],
                lane_width,
                (s,c)
            )
        )
        # R-10 - south to north
        net.add_lane(
            "I-9:n-in",
            "I-8:s-out",
            StraightLane(
                [56, 100],
                [56, 92],
                lane_width,
                (n,c)
            )
        )
        
        
        """Intersection 8"""
        # I-8 - north to south
        net.add_lane(
            "I-8:n-out",
            "I-8:s-in",
            StraightLane(
                [52, 80],
                [52, 92],
                lane_width,
                (s,n)
            )
        )
        # I-8 - north to west
        net.add_lane(
            "I-8:n-out",
            "I-8:s-in",
            self._build_circlelane(
                [52, 80],
                [48, 84],
                right_turn,
                0,
                90,
                3,
                (n,c)
            )
        )
        
        # I-8 - south to north
        net.add_lane(
            "I-8:s-out",
            "I-8:n-in",
            StraightLane(
                [56, 92],
                [56, 80],
                lane_width,
                (n,c)
            )
        )
        # I-8 - south to west
        net.add_lane(
            "I-8:s-out",
            "I-8:w-in",
            self._build_circlelane(
                [56, 92],
                [48, 84],
                left_turn,
                0,
                -90,
                2,
                (n,n)
            )
        )
        
        # I-8 - west to north
        net.add_lane(
            "I-8:w-out",
            "I-8:n-in",
            self._build_circlelane(
                [48, 88],
                [56, 80],
                left_turn,
                90,
                0,
                0,
                (n,n)
            )
        )
        # I-8 - west to south
        net.add_lane(
            "I-8:w-out",
            "I-8:n-in",
            self._build_circlelane(
                [48, 88],
                [52, 92],
                right_turn,
                -90,
                0,
                0,
                (n,c)
            )
        )
        
        
        """Road 12"""
        # R-12 - north to south
        net.add_lane(
            "T-3:s-in",
            "I-8:n-out",
            StraightLane(
                [52, 50],
                [52, 80],
                lane_width,
                (s,c)
            )
        )
        # R-12 - south to north
        net.add_lane(
            "I-8:n-in",
            "T-3:s-out",
            StraightLane(
                [56, 80],
                [56, 50],
                lane_width,
                (n,c)
            )
        )
        

        """Turn 3"""
        # T-3 south to east
        net.add_lane(
            "T-3:s-out",
            "T-3:e-in",
            self._build_circlelane(
                [56, 50],
                [60, 46],
                right_turn,
                -180,
                -90,
                0,
                (s,c)
            )
        )
        # T-3 east to south
        net.add_lane(
            "T-3:e-out",
            "T-3:s-in",
            self._build_circlelane(
                [60, 42],
                [52, 50],
                left_turn,
                -90,
                -180,
                0,
                (n,c)
            )
        )
        
        
        """Road 13"""
        # R-13 - east to west
        net.add_lane(
            "I-8:w-in",
            "I-10:e-out",
            StraightLane(
                [48, 84],
                [6, 84],
                lane_width,
                (s,c)
            )
        )
        # R-13 - west to east
        net.add_lane(
            "I-10:e-in",
            "I-8:w-out",
            StraightLane(
                [6, 88],
                [48, 88],
                lane_width,
                (n,c)
            )
        )
        
        
        """Intersection 10"""
        # I-10 - east to west
        net.add_lane(
            "I-10:e-out",
            "I-10:w-in",
            StraightLane(
                [6, 84],
                [-6, 84],
                lane_width,
                (s,n),
                priority=3
            )
        )
        # I-10 - east to north
        net.add_lane(
            "I-10:e-out",
            "I-10:n-in",
            self._build_circlelane(
                [6, 84],
                [2, 80],
                right_turn,
                90,
                180,
                3,
                (n,c)
            )
        )
        
        # I-10 - north to west
        net.add_lane(
            "I-10:n-out",
            "I-10:w-in",
            self._build_circlelane(
                [-2, 80],
                [-6, 84],
                right_turn,
                0,
                90,
                0,
                (n,c)
            )
        )
        # I-10 - north to east
        net.add_lane(
            "I-10:n-out",
            "I-10:e-in",
            self._build_circlelane(
                [-2, 80],
                [6, 88],
                left_turn,
                180,
                90,
                0,
                (n,n)
            )
        )
        
        # I-10 - west to east
        net.add_lane(
            "I-10:w-out",
            "I-10:e-in",
            StraightLane(
                [-6, 88],
                [6, 88],
                lane_width,
                (n,c),
                priority=3
            )
        )
        # I-10 - west to north
        net.add_lane(
            "I-10:w-out",
            "I-10:n-in",
            self._build_circlelane(
                [-6, 88],
                [2, 80],
                left_turn,
                90,
                0,
                2,
                (n,n)
            )
        )
        
        
        """Road 14"""
        # R-14 - east to west
        net.add_lane(
            "I-10:w-in",
            "T-6:e-out",
            StraightLane(
                [-6, 84],
                [-75, 84],
                lane_width,
                (s,c)
            )
        )
        # R-14 - west to east
        net.add_lane(
            "T-6:e-in",
            "I-10:w-out",
            StraightLane(
                [-75, 88],
                [-6, 88],
                lane_width,
                (n,c)
            )
        )
        
        
        """Turn 6"""
        # T-6 - east to north
        net.add_lane(
            "T-6:e-out",
            "T-6:n-in",
            self._build_circlelane(
                [-75, 84],
                [-79, 80],
                right_turn,
                90,
                180,
                0,
                (s,c)
            )
        )
        # T-6 - north to east
        net.add_lane(
            "T-6:n-out",
            "T-6:e-in",
            self._build_circlelane(
                [-83, 80],
                [-75, 88],
                left_turn,
                180,
                90,
                0,
                (n,c)
            )
        )
        
        
        """Road 15"""
        # R-15 - north to south
        net.add_lane(
            "T-5:s-in",
            "T-6:n-out",
            StraightLane(
                [-83, 45],
                [-83, 80],
                lane_width,
                (s,c)
            )
        )
        # R-15 - south to north
        net.add_lane(
            "T-6:n-in",
            "T-5:s-out",
            StraightLane(
                [-79, 80],
                [-79, 45],
                lane_width,
                (n,c)
            )
        )
        
        
        """Turn 5"""
        # T-5 - south to west
        net.add_lane(
            "T-5:s-out",
            "T-5:w-in",
            self._build_circlelane(
                [-79, 45],
                [-87, 37],
                left_turn,
                0,
                -90,
                0,
                (s,c)
            )
        )
        # T-5 - west to south
        net.add_lane(
            "T-5:w-out",
            "T-5:s-in",
            self._build_circlelane(
                [-87, 41],
                [-83, 45],
                right_turn,
                -90,
                0,
                0,
                (n,c)
            )
        )
        
        
        """Road 16"""
        # R-16 - east to west
        net.add_lane(
            "T-5:w-in",
            "I-11:e-out",
            StraightLane(
                [-87, 37],
                [-116, 37],
                lane_width,
                (s,c)
            )
        )
        # R-16 - west to east
        net.add_lane(
            "I-11:e-in",
            "T-5:w-out",
            StraightLane(
                [-116, 41],
                [-87, 41],
                lane_width,
                (n,c)
            )
        )
        
        
        """Intersection 11"""
        # I-11 - east to west
        net.add_lane(
            "I-11:e-out",
            "I-11:w-in",
            StraightLane(
                [-116, 37],
                [-128, 37],
                lane_width,
                (s,n),
                priority=3
            )
        )
        # I-11 - east to north
        net.add_lane(
            "I-11:e-out",
            "I-11:n-in",
            self._build_circlelane(
                [-116, 37],
                [-120, 33],
                right_turn,
                90,
                180,
                3,
                (n,c),
            )
        )

        # I-11 - north to west
        net.add_lane(
            "I-11:n-out",
            "I-11:w-in",
            self._build_circlelane(
                [-124, 33],
                [-128, 37],
                right_turn,
                0,
                90,
                0,
                (n,c)
            )
        )
        # I-11 - north to east
        net.add_lane(
            "I-11:n-out",
            "I-11:e-in",
            self._build_circlelane(
                [-124, 33],
                [-120, 41],
                left_turn,
                180,
                90,
                0,
                (n,n)
            )
        )
        
        # I-11 - west to east
        net.add_lane(
            "I-11:w-out",
            "I-11:e-in",
            StraightLane(
                [-128, 41],
                [-116, 41],
                lane_width,
                (s,c),
                priority=3
            )
        )
        # I-11 - west to north
        net.add_lane(
            "I-11:w-out",
            "I-11:n-in",
            self._build_circlelane(
                [-128, 41],
                [-120, 33],
                left_turn,
                90,
                0,
                2,
                (n,n)
            )
        )
        
        
        """Road 17"""
        # R-17 - south to north
        net.add_lane(
            "I-11:n-in",
            "I-12:s-out",
            StraightLane(
                [-120, 33],
                [-120, 6],
                lane_width,
                (s,c)
            )
        )
        # R-17 - north to south
        net.add_lane(
            "I-12:s-in",
            "I-11:n-out",
            StraightLane(
                [-124, 6],
                [-124, 33],
                lane_width,
                (n,c)
            )
        )
        
        
        """Exit 8"""
        # E-8 - east to west
        net.add_lane(
            "I-11:w-in",
            "E-8:out",
            StraightLane(
                [-128, 37],
                [-170, 37],
                lane_width,
                (s,c)
            )
        )
        # E-8 - west to east
        net.add_lane(
            "E-8:in",
            "I-11:w-out",
            StraightLane(
                [-128, 41],
                [-170, 41],
                lane_width,
                (c,n)
            )
        )
        # E-8 - turn car around
        net.add_lane(
            "E-8:out",
            "E-8:in",
            CircularLane(
                [-170, 41],
                4,
                np.deg2rad(-90),
                np.deg2rad(-270),
                left_turn,
                lane_width,
                (n, c)
            )
        )
        
        
        
        
        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        # ego_vehicle = self.action_type.vehicle_class(
        #     road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=30
        # )
        # road.vehicles.append(ego_vehicle)

        # other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # for position, speed in [(90, 29), (70, 31), (5, 31.5)]:
        #     lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
        #     position = lane.position(position + self.np_random.uniform(-5, 5), 0)
        #     speed += self.np_random.uniform(-1, 1)
        #     road.vehicles.append(other_vehicles_type(road, position, speed=speed))


        # ego_lane = self.road.network.get_lane(("wxs", "I-12:e-out", 0))
        # ego_lane = self.road.network.get_lane(("T-5:w-in", "I-11:e-out", 0))
        ego_lane = self.road.network.get_lane(("I-3:w-in", "T-2:e-out", 0))
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(20, 0),
            speed=0,
            heading=ego_lane.heading_at(-90), # Use this to change the direction of the car. "0" is north
        )
        # ego_vehicle.target_speed = 5
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
                
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }
    
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]