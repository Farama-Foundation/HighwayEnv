from __future__ import annotations
import math

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork, WeightedRoadnetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.utils import Vector
from highway_env.road.lanes.lane_utils import LaneType, LineType
from highway_env.road.lanes.abstract_lanes import AbstractLane

from highway_env.road.regulation import RegulatedRoad
from highway_env.network_builder import NetworkBuilder, StraightPath, CircularPath, Path
from highway_env.road.lanes.unweighted_lanes import StraightLane, SineLane, CircularLane


class CarpetCity(AbstractEnv):
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
                    "target_speeds": [-10, 0, 5, 10, 20] #np.linspace(0, 30, 10)
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
    
    def _make_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""

        nb = NetworkBuilder()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        roundabout_radius = 20 # [m]
        left_turn = False
        right_turn = True
        lane_width = AbstractLane.DEFAULT_WIDTH
        turn_radius = 20
        
        
        
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        alpha = 24  # [deg]

        net = WeightedRoadnetwork()
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
                50,
                LaneType.ROUNDABOUT
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
                50,
                LaneType.ROUNDABOUT
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
                50,
                LaneType.ROUNDABOUT
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
                50,
                LaneType.ROUNDABOUT
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
                50,
                LaneType.ROUNDABOUT
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
                50,
                LaneType.ROUNDABOUT
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
                50,
                LaneType.ROUNDABOUT
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
                50,
                LaneType.ROUNDABOUT
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
            "I-10:n-out",
            "ses",
            StraightLane(
                [2, 150],
                [2, dev / 2],
                line_types=(s, c),
                priority=3
            ),
            50,
            LaneType.ROAD
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
            50,
            LaneType.ROUNDABOUT
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
            50,
            LaneType.ROUNDABOUT
        )
        # South Exit straight lane
        net.add_lane(
            "sxs",
            "I-10:n-in",
            StraightLane(
                [-2, dev / 2],
                [-2, 150],
                line_types=(n, c),
                priority=3
            ),
            50,
            LaneType.ROAD
        )

        # East Enter straight lane
        net.add_lane(
            "I-4:w-out",
            "ees",
            StraightLane(
                [300, -2],
                [dev / 2, -2],
                line_types=(s, c),
                priority=3
            ),
            50,
            LaneType.ROAD
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
            50,
            LaneType.ROUNDABOUT
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
            50,
            LaneType.ROUNDABOUT
        )
        # East Exit straight lane
        net.add_lane(
            "exs",
            "I-4:w-in",
            StraightLane(
                [dev / 2, 2],
                [300, 2],
                line_types=(n, c),
                priority=3
            ),
            50,
            LaneType.ROAD
        )

        # North Enter straight lane
        net.add_lane(
            "I-1:s-out",
            "nes",
            StraightLane(
                [-2, -200],
                [-2, -dev / 2],
                line_types=(s, c),
                priority=3
            ),
            50,
            LaneType.ROAD
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
            50,
            LaneType.ROUNDABOUT
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
            50,
            LaneType.ROUNDABOUT
        )
        # North Exit straight lane
        net.add_lane(
            "nxs",
            "I-1:s-in",
            StraightLane(
                [2, -dev / 2],
                [2, -200],
                line_types=(n, c),
                priority=3
            ),
            50,
            LaneType.ROAD
        )

        # West Enter straight lane
        net.add_lane(
            "I-12:e-out",
            "wes",
            StraightLane(
                [-116, 2],
                [-dev / 2, 2],
                line_types=(s, c),
                priority=3
            ),
            50,
            LaneType.ROAD
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
            50,
            LaneType.ROUNDABOUT
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
            50,
            LaneType.ROUNDABOUT
        )
        # West Exit straight lane
        net.add_lane(
            "wxs",
            "I-12:e-in",
            StraightLane(
                [-dev / 2, -2],
                [-116, -2],
                line_types=(n, c),
                priority=3
            ),
            50,
            LaneType.ROAD
        )
        
        """
        Nodes comming out of the roundabout
        
        south exit -> "I-10:n-in" : [-2, 150]
        north exit -> "I-1:s-in"  : [-2, -200]
        east  exit -> "I-4:w-in"  : [300, 2]
        west  exit -> "I-12:e-in" : [-116, -2]
        
        """
        
        
        """Intersection 12"""
        nb.add_intersection(
            "I-12",
            {
                nb.CardinalDirection.SOUTH : [-120, 6],
                nb.CardinalDirection.WEST  : [-128, 2],
                nb.CardinalDirection.EAST  : [-116, -2],
            },
            nb.PathPriority.EAST_WEST,
            20
        )
        
        """Road 18"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-12:w-out", "T-1:e-in", (s,c), 40, LaneType.ROAD),
                StraightPath("T-1:e-out", "I-12:w-in", (n,c), 40, LaneType.ROAD),
            ]
        })
        
        """Turn 1"""
        nb.add_multiple_nodes({
            "T-1:e-in"  : [-300, -2],
            "T-1:e-out" : [-300, 2],
            "T-1:n-in"  : [-324, -22],
            "T-1:n-out" : [-320, -22],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-1:e-in", "T-1:n-out", -90, turn_radius, right_turn, (s,c), 20, LaneType.ROAD),
                CircularPath("T-1:n-in", "T-1:e-out", 180, turn_radius + lane_width, left_turn, (n,c), 20, LaneType.ROAD)
            ]
        })
        
        
        """Road 19"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("R-1:s-out", "T-1:n-in", (s,c), 50, LaneType.ROAD),
                StraightPath("T-1:n-out", "R-1:s-in", (n,c), 50, LaneType.ROAD),
            ]
        })
        
        """Roundabout 1"""
        nb.add_roundabout(
            "R-1",
            {
                nb.CardinalDirection.SOUTH : [-320, -180],
                nb.CardinalDirection.WEST  : [-348, -204],
                nb.CardinalDirection.EAST  : [-296, -208],
            },
            30
        )
        
        
        """Road 1"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("R-1:e-out", "I-1:w-in", (s,c), 60, LaneType.ROAD),
                StraightPath("I-1:w-out", "R-1:e-in", (n,c), 60, LaneType.ROAD),
            ]
        })
        
        
        """Intersection 1"""
        nb.add_intersection(
            "I-1",
            {
                nb.CardinalDirection.NORTH : [-2, -212],
                nb.CardinalDirection.SOUTH : [2, -200],
                nb.CardinalDirection.WEST  : [-6, -204],
                nb.CardinalDirection.EAST  : [6, -208],
            },
            nb.PathPriority.EAST_WEST,
            40
        )
        
        """Road 2"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-1:e-out", "I-2:w-in", (s,c), 60, LaneType.ROAD),
                StraightPath("I-2:w-out", "I-1:e-in", (n,c), 60, LaneType.ROAD),
            ]
        })
        
        
        """Intersection 2"""
        nb.add_intersection(
            "I-2",
            {
                nb.CardinalDirection.NORTH : [54, -212],
                nb.CardinalDirection.SOUTH : [58, -200],
                nb.CardinalDirection.WEST  : [50, -204],
            },
            nb.PathPriority.NORTH_SOUTH,
            30
        )
        
        
        """Road 3"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-2:s-out", "T-2:n-in", (s,c), 60, LaneType.ROAD),
                StraightPath("T-2:n-out", "I-2:s-in", (n,c), 60, LaneType.ROAD),
            ]
        })
        
        
        """Turn 2"""
        nb.add_multiple_nodes({
            "T-2:e-in"  : [78, -80],
            "T-2:e-out" : [78, -76],
            "T-2:n-in"  : [54, -100],
            "T-2:n-out" : [58, -100],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-2:e-in", "T-2:n-out", -90, turn_radius, right_turn, (s,c), 20, LaneType.ROAD),
                CircularPath("T-2:n-in", "T-2:e-out", 180, turn_radius + lane_width, left_turn, (n,c), 20, LaneType.ROAD)
            ]
        })
        
        
        """Road 4"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-3:w-out", "T-2:e-in", (s,c), 40, LaneType.ROAD),
                StraightPath("T-2:e-out", "I-3:w-in", (n,c), 40, LaneType.ROAD),
            ]
        })
        
        
        """Intersection 3"""
        nb.add_intersection(
            "I-3",
            {
                nb.CardinalDirection.SOUTH : [308, -72],
                nb.CardinalDirection.WEST  : [300, -76],
                nb.CardinalDirection.EAST  : [312, -80],
            },
            nb.PathPriority.EAST_WEST,
            20
        )
        
        
        """Road 5"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-3:s-out", "I-4:n-in", (s,c), 60, LaneType.ROAD),
                StraightPath("I-4:n-out", "I-3:s-in", (n,c), 60, LaneType.ROAD),
            ]
        })


        """Intersection 4"""
        nb.add_intersection(
            "I-4",
            {
                nb.CardinalDirection.NORTH : [304, -6],
                nb.CardinalDirection.SOUTH : [308, 6],
                nb.CardinalDirection.WEST  : [300, 2],
            },
            nb.PathPriority.NORTH_SOUTH,
            30
        )
        
        
        """Road 6"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-4:s-out", "I-5:n-in", (s,c), 20, LaneType.ROAD),
                StraightPath("I-5:n-out", "I-4:s-in", (n,c), 20, LaneType.ROAD),
            ]
        })
        
        
        """Intersection 5"""
        nb.add_intersection(
            "I-5",
            {
                nb.CardinalDirection.NORTH : [304, 50],
                nb.CardinalDirection.SOUTH : [308, 62],
                nb.CardinalDirection.EAST  : [312, 54],
            },
            nb.PathPriority.NORTH_SOUTH,
            30
        )
        
        
        """Road 7"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-5:s-out", "I-6:n-in", (s,c), 20, LaneType.ROAD),
                StraightPath("I-6:n-out", "I-5:s-in", (n,c), 20, LaneType.ROAD),
            ]
        })


        """Intersection 6"""
        nb.add_intersection(
            "I-6",
            {
                nb.CardinalDirection.NORTH : [304, 100],
                nb.CardinalDirection.SOUTH : [308, 112],
                nb.CardinalDirection.WEST  : [300, 108],
            },
            nb.PathPriority.NORTH_SOUTH,
            30
        )
        
        
        """Road 11"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-6:w-out", "T-3:e-in", (s,c), 20, LaneType.ROAD),
                StraightPath("T-3:e-out", "I-6:w-in", (n,c), 20, LaneType.ROAD),
            ]
        })
        
        
        """Turn 3"""
        nb.add_multiple_nodes({
            "T-3:e-in"  : [228, 104],
            "T-3:e-out" : [228, 108],
            "T-3:s-in"  : [208, 128],
            "T-3:s-out" : [204, 128],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-3:e-in", "T-3:s-out", -90, turn_radius + lane_width, left_turn, (s,c), 20, LaneType.ROAD),
                CircularPath("T-3:s-in", "T-3:e-out", 0, turn_radius, right_turn, (n,c), 20, LaneType.ROAD)
            ]
        })
        
        
        """Road 12"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-3:s-out", "I-8:n-in", (s,c), 20, LaneType.ROAD),
                StraightPath("I-8:n-out", "T-3:s-in", (n,c), 20, LaneType.ROAD),
            ]
        })
        
        
        """Intersection 8"""
        nb.add_intersection(
            "I-8",
            {
                nb.CardinalDirection.NORTH : [204, 150],
                nb.CardinalDirection.SOUTH : [208, 162],
                nb.CardinalDirection.WEST  : [200, 158],
            },
            nb.PathPriority.NORTH_SOUTH,
            30
        )
        
        
        """Road 13"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-8:w-out", "I-10:e-in", (s,c), 20, LaneType.ROAD),
                StraightPath("I-10:e-out", "I-8:w-in", (n,c), 20, LaneType.ROAD),
            ]
        })
        
        
        """Intersection 10"""
        nb.add_intersection(
            "I-10",
            {
                nb.CardinalDirection.NORTH : [-2, 150],
                nb.CardinalDirection.WEST  : [-6, 158],
                nb.CardinalDirection.EAST  : [6, 154],
            },
            nb.PathPriority.EAST_WEST,
            30
        )
        
        
        """Road 10"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-8:s-out", "I-9:n-in", (s,c), 20, LaneType.ROAD),
                StraightPath("I-9:n-out", "I-8:s-in", (n,c), 20, LaneType.ROAD),
            ]
        })


        """Intersection 9"""
        nb.add_intersection(
            "I-9",
            {
                nb.CardinalDirection.NORTH : [204, 300],
                nb.CardinalDirection.SOUTH : [208, 312],
                nb.CardinalDirection.EAST  : [212, 304],
            },
            nb.PathPriority.NORTH_SOUTH,
            30
        )


        """Road 9"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-9:e-out", "I-7:w-in", (s,c), 20, LaneType.ROAD),
                StraightPath("I-7:w-out", "I-9:e-in", (n,c), 20, LaneType.ROAD),
            ]
        })
        

        """Intersection 7"""
        nb.add_intersection(
            "I-7",
            {
                nb.CardinalDirection.NORTH : [304, 300],
                nb.CardinalDirection.WEST  : [300, 308],
                nb.CardinalDirection.EAST  : [312, 304],
            },
            nb.PathPriority.EAST_WEST,
            30
        )
        
        
        """Road 8"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-6:s-out", "I-7:n-in", (s,c), 20, LaneType.ROAD),
                StraightPath("I-7:n-out", "I-6:s-in", (n,c), 20, LaneType.ROAD),
            ]
        })
        
        
        """Road 14"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-10:w-out", "T-4:e-in", (s,c), 20, LaneType.ROAD),
                StraightPath("T-4:e-out", "I-10:w-in", (n,c), 20, LaneType.ROAD),
            ]
        })
        
        
        nb.add_multiple_nodes({
            "T-1:e-in"  : [-300, -2],
            "T-1:e-out" : [-300, 2],
            "T-1:n-in"  : [-324, -22],
            "T-1:n-out" : [-320, -22],
        })
        
        """Turn 4"""
        nb.add_multiple_nodes({
            "T-4:e-in"  : [-20, 154],
            "T-4:e-out" : [-20, 158],
            "T-4:n-in"  : [-44, 134],
            "T-4:n-out" : [-40, 134],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-4:e-in", "T-4:n-out", -90, turn_radius, right_turn, (s,c), 20, LaneType.ROAD),
                CircularPath("T-4:n-in", "T-4:e-out", 180, turn_radius + lane_width, left_turn, (n,c), 20, LaneType.ROAD)
            ]
        })
        


        """Road 15"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-4:n-out", "T-5:s-in", (s,c), 20, LaneType.ROAD),
                StraightPath("T-5:s-out", "T-4:n-in", (n,c), 20, LaneType.ROAD),
            ]
        })


        """Turn 5"""
        nb.add_multiple_nodes({
            "T-5:w-in"  : [-60, 88],
            "T-5:w-out" : [-60, 84],
            "T-5:s-in"  : [-40, 108],
            "T-5:s-out" : [-44, 108],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-5:w-in", "T-5:s-out", 90, turn_radius, right_turn, (s,c), 20, LaneType.ROAD),
                CircularPath("T-5:s-in", "T-5:w-out", 0, turn_radius + lane_width, left_turn, (n,c), 20, LaneType.ROAD)
            ]
        })


        """Road 16"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-11:e-out", "T-5:w-in", (s,c), 20, LaneType.ROAD),
                StraightPath("T-5:w-out", "I-11:e-in", (n,c), 20, LaneType.ROAD),
            ]
        })


        """Intersection 11"""
        nb.add_intersection(
            "I-11",
            {
                nb.CardinalDirection.NORTH : [-124, 80],
                nb.CardinalDirection.WEST  : [-128, 88],
                nb.CardinalDirection.EAST  : [-116, 84],
            },
            nb.PathPriority.EAST_WEST,
            30
        )


        """Road 17"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-12:s-out", "I-11:n-in", (s,c), 20, LaneType.ROAD),
                StraightPath("I-11:n-out", "I-12:s-in", (n,c), 20, LaneType.ROAD),
            ]
        })
        
        
        nb.build_paths(net)
        
        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        self.road = road
        return

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        road = self.road

        # ego_lane = self.road.network.get_lane(("wxs", "I-12:e-in", 0))
        # ego_lane = self.road.network.get_lane(("R-1:e-out", "I-1:w-in", 0))
        # ego_lane = self.road.network.get_lane(("exs", "I-4:w-in", 0))
        # ego_lane = self.road.network.get_lane(("I-2:s-out", "T-2:n-in", 0))
        # ego_lane = self.road.network.get_lane(("I-4:s-out", "I-5:n-in", 0))
        # ego_lane = self.road.network.get_lane(("I-6:w-out", "T-3:e-in", 0))
        # ego_lane = self.road.network.get_lane(("I-10:w-out", "T-4:e-in", 0))
        # ego_lane = self.road.network.get_lane(("I-12:s-out", "I-11:n-in", 0))
        ego_lane = self.road.network.get_lane(("T-1:n-out", "R-1:s-in", 0))
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(0, 0),            # Use the first value to place car down the road
            speed=0,                            # Speed of car
            heading=ego_lane.heading_at(90),   # Use this to change the direction of the car. "0" is north
        )

        # ego_vehicle.plan_route_to("sxr")
        
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        
    # Note this reward function is just generic from another template
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

    # Note this reward function is just generic from another template
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