from __future__ import annotations

import copy
import logging
import os
import pickle
import random
import time

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.envs.weighted_utils import WeightedUtils
from highway_env.network_builder import CircularPath, NetworkBuilder, Path, StraightPath
from highway_env.road.lanes.abstract_lanes import AbstractLane
from highway_env.road.lanes.lane_utils import LaneType, LineType
from highway_env.road.lanes.unweighted_lanes import CircularLane, SineLane, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import Road, RoadNetwork, WeightedRoadnetwork
from highway_env.utils import Vector, near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


# Debug logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("carpet_city.log")
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Episode and route logger
routes_logger = logging.getLogger(__name__ + ".routes")
routes_logger.setLevel(logging.INFO)
file_handler_routes = logging.FileHandler("carpet_city_routes.log")
routes_formatter = logging.Formatter("%(message)s")
file_handler_routes.setFormatter(routes_formatter)
routes_logger.addHandler(file_handler_routes)
routes_logger.propagate = False


class CarpetCity(AbstractEnv, WeightedUtils):
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
                    "target_speeds": np.linspace(0, 40, 10),
                },
                "simulation_frequency": 15,
                "lanes_count": 2,
                "vehicles_count": 30,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 120,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -10,  # The reward received when colliding with a vehicle.
                # zero for other lanes.
                "high_speed_reward": 1,
                "distance_from_goal": 2,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "headway_evaluation": 0.5,
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
                "screen_height": 600,
                "screen_width": 1200,
            }
        )
        return config

    def calculate_shortest_paths(self):
        file_name = "carpet-city-paths.pkl"

        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                routes = pickle.load(f)
        else:
            routes = {}

        edges = self.road.network.graph_net.edges

        total_edges = len(edges) * (len(edges) - 1)
        edges_left = total_edges

        for _, s, _ in edges:
            if s in routes:
                edges_left -= len(routes[s])

            edges_left -= 1

        if edges_left < 0:
            edges_left = 0

        avg_time = None
        count = 0

        i = 0
        for _, startpoint, _ in edges:
            # shortest_path_logger.info(f"---::: {i}/{len(edges)-1} @ Beginning on '{startpoint}' :::---")
            print(f"---::: {i}/{len(edges)} @ Beginning on '{startpoint}' :::---")

            if startpoint not in routes:
                routes[startpoint] = {}

            j = 0
            for _, destination, _ in edges:
                if startpoint == destination:
                    j += 1
                    continue

                start_t = time.time()
                if destination in routes[startpoint]:
                    # shortest_path_logger.info(f"\t$$$ Skipping already calculated route from '{startpoint}' ~> '{destination}'\t @ path is: {routes[startpoint][destination]}")
                    print(
                        f"\t$$$ Skipping already calculated route from '{startpoint}' ~> '{destination}'\t"
                    )

                    j += 1
                    edges_left -= 1
                    continue

                result = self.road.network.shortest_path(startpoint, destination)

                routes[startpoint][destination] = result

                with open(file_name, "wb") as f:
                    pickle.dump(routes, f, protocol=pickle.HIGHEST_PROTOCOL)

                end_t = time.time()
                duration = end_t - start_t

                if avg_time is None:
                    avg_time = duration
                else:
                    avg_time = (avg_time * count + duration) / (count + 1)

                count += 1
                edges_left -= 1

                if avg_time is not None and edges_left > 0:
                    time_left = avg_time * edges_left
                    hrs = int(time_left // 3600)
                    mins = int((time_left % 3600) // 60)
                    secs = int(time_left % 60)
                    time_left_str = f"{hrs}h {mins}m {secs}s"
                else:
                    time_left_str = "N/A"

                # shortest_path_logger.info(f"\t$$$ start: {i}/{len(edges)-1} @ dest.: {j}/{len(edges)-2} @ '{startpoint}' ~> '{destination}'")
                print(
                    f"\t$$$ estimate time left: {time_left_str}\t $$$ start: {i}/{len(edges)-1} @ dest.: {j}/{len(edges)-2} @ '{startpoint}' ~> '{destination}'"
                )

                j += 1

            i += 1
            edges_left -= 1

        print(f"Routes have been calculated and saved to {file_name}")

    def _reset(self) -> None:
        if not hasattr(self, "episode_count"):
            self.episode_count = 0

        logger.info(f"Episode: {self.episode_count}")
        routes_logger.info(f"Episode: {self.episode_count}")

        self._make_road()

        # self.calculate_shortest_paths()
        # return

        if not hasattr(self, "local_graph_net"):
            self.local_graph_net = copy.deepcopy(self.road.network.graph_net)

        if not hasattr(self, "shortest_paths"):
            with open("carpet-city-paths.pkl", "rb") as f:
                self.shortest_paths = pickle.load(f)

        if not hasattr(self, "has_been_categorized"):
            self._categorize_edges_by_type()

        self._make_vehicles(self.config["vehicles_count"])
        self.episode_count += 1

    def _make_road(self):
        """Create a road composed of straight adjacent lanes."""

        nb = NetworkBuilder()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        roundabout_radius = 20  # [m]
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
                LaneType.ROUNDABOUT,
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
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(107.5, 50),
            LaneType.CITY_ROAD,
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
                speed_limit=40,
            ),
            nb.get_weight(18, 50),
            LaneType.ROUNDABOUT,
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
                speed_limit=40,
            ),
            nb.get_weight(18, 50),
            LaneType.ROUNDABOUT,
        )
        # South Exit straight lane
        net.add_lane(
            "sxs",
            "I-10:n-in",
            StraightLane(
                [-2, dev / 2],
                [-2, 150],
                line_types=(n, c),
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(107.5, 50),
            LaneType.CITY_ROAD,
        )

        # East Enter straight lane
        net.add_lane(
            "I-4:w-out",
            "ees",
            StraightLane(
                [300, -2],
                [dev / 2, -2],
                line_types=(s, c),
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(107.5, 50),
            LaneType.CITY_ROAD,
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
                speed_limit=40,
            ),
            nb.get_weight(18, 50),
            LaneType.ROUNDABOUT,
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
                speed_limit=40,
            ),
            nb.get_weight(18, 50),
            LaneType.ROUNDABOUT,
        )
        # East Exit straight lane
        net.add_lane(
            "exs",
            "I-4:w-in",
            StraightLane(
                [dev / 2, 2],
                [300, 2],
                line_types=(n, c),
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(107.5, 50),
            LaneType.CITY_ROAD,
        )

        # North Enter straight lane
        net.add_lane(
            "I-1:s-out",
            "nes",
            StraightLane(
                [-2, -200],
                [-2, -dev / 2],
                line_types=(s, c),
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(107.5, 50),
            LaneType.CITY_ROAD,
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
                speed_limit=40,
            ),
            nb.get_weight(18, 50),
            LaneType.ROUNDABOUT,
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
                speed_limit=40,
            ),
            nb.get_weight(18, 50),
            LaneType.ROUNDABOUT,
        )
        # North Exit straight lane
        net.add_lane(
            "nxs",
            "I-1:s-in",
            StraightLane(
                [2, -dev / 2],
                [2, -200],
                line_types=(n, c),
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(107.5, 50),
            LaneType.CITY_ROAD,
        )

        # West Enter straight lane
        net.add_lane(
            "I-12:e-out",
            "wes",
            StraightLane(
                [-116, 2],
                [-dev / 2, 2],
                line_types=(s, c),
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(107.5, 50),
            LaneType.CITY_ROAD,
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
                speed_limit=40,
            ),
            nb.get_weight(18, 50),
            LaneType.ROUNDABOUT,
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
                speed_limit=40,
            ),
            nb.get_weight(18, 50),
            LaneType.ROUNDABOUT,
        )
        # West Exit straight lane
        net.add_lane(
            "wxs",
            "I-12:e-in",
            StraightLane(
                [-dev / 2, -2],
                [-116, -2],
                line_types=(n, c),
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(107.5, 50),
            LaneType.CITY_ROAD,
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
                nb.CardinalDirection.SOUTH: [-120, 6],
                nb.CardinalDirection.WEST: [-128, 2],
                nb.CardinalDirection.EAST: [-116, -2],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50),
        )

        """Road 18"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-12:w-out",
                        "T-1:e-in",
                        (s, c),
                        nb.get_weight(172, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "T-1:e-out",
                        "I-12:w-in",
                        (n, c),
                        nb.get_weight(172, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Turn 1"""
        nb.add_multiple_nodes(
            {
                "T-1:e-in": [-300, -2],
                "T-1:e-out": [-300, 2],
                "T-1:n-in": [-324, -22],
                "T-1:n-out": [-320, -22],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.CIRCULAR: [
                    CircularPath(
                        "T-1:e-in",
                        "T-1:n-out",
                        -90,
                        turn_radius,
                        right_turn,
                        (s, c),
                        nb.get_weight(31, 50),
                        LaneType.CITY_ROAD,
                    ),
                    CircularPath(
                        "T-1:n-in",
                        "T-1:e-out",
                        180,
                        turn_radius + lane_width,
                        left_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Road 19"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "R-1:s-out",
                        "T-1:n-in",
                        (s, c),
                        nb.get_weight(158, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "T-1:n-out",
                        "R-1:s-in",
                        (n, c),
                        nb.get_weight(158, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Roundabout 1"""
        nb.add_roundabout(
            "R-1",
            {
                nb.CardinalDirection.SOUTH: [-320, -180],
                nb.CardinalDirection.WEST: [-348, -204],
                nb.CardinalDirection.EAST: [-296, -208],
            },
            nb.get_weight(12, 50),
        )

        """Road 1"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "R-1:e-out",
                        "I-1:w-in",
                        (s, c),
                        nb.get_weight(290, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "I-1:w-out",
                        "R-1:e-in",
                        (n, c),
                        nb.get_weight(290, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Intersection 1"""
        nb.add_intersection(
            "I-1",
            {
                nb.CardinalDirection.NORTH: [-2, -212],
                nb.CardinalDirection.SOUTH: [2, -200],
                nb.CardinalDirection.WEST: [-6, -204],
                nb.CardinalDirection.EAST: [6, -208],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50),
        )

        """Road 2"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-1:e-out",
                        "I-2:w-in",
                        (s, c),
                        nb.get_weight(54, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "I-2:w-out",
                        "I-1:e-in",
                        (n, c),
                        nb.get_weight(54, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Intersection 2"""
        nb.add_intersection(
            "I-2",
            {
                nb.CardinalDirection.NORTH: [54, -212],
                nb.CardinalDirection.SOUTH: [58, -200],
                nb.CardinalDirection.WEST: [50, -204],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 50),
        )

        """Road 3"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-2:s-out",
                        "T-2:n-in",
                        (s, c),
                        nb.get_weight(100, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "T-2:n-out",
                        "I-2:s-in",
                        (n, c),
                        nb.get_weight(100, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Turn 2"""
        nb.add_multiple_nodes(
            {
                "T-2:e-in": [78, -80],
                "T-2:e-out": [78, -76],
                "T-2:n-in": [54, -100],
                "T-2:n-out": [58, -100],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.CIRCULAR: [
                    CircularPath(
                        "T-2:e-in",
                        "T-2:n-out",
                        -90,
                        turn_radius,
                        right_turn,
                        (s, c),
                        nb.get_weight(31, 50),
                        LaneType.CITY_ROAD,
                    ),
                    CircularPath(
                        "T-2:n-in",
                        "T-2:e-out",
                        180,
                        turn_radius + lane_width,
                        left_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Road 4"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-3:w-out",
                        "T-2:e-in",
                        (s, c),
                        nb.get_weight(222, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "T-2:e-out",
                        "I-3:w-in",
                        (n, c),
                        nb.get_weight(222, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Intersection 3"""
        nb.add_intersection(
            "I-3",
            {
                nb.CardinalDirection.SOUTH: [308, -72],
                nb.CardinalDirection.WEST: [300, -76],
                nb.CardinalDirection.EAST: [312, -80],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50),
        )

        """Road 5"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-3:s-out",
                        "I-4:n-in",
                        (s, c),
                        nb.get_weight(66, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "I-4:n-out",
                        "I-3:s-in",
                        (n, c),
                        nb.get_weight(66, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Intersection 4"""
        nb.add_intersection(
            "I-4",
            {
                nb.CardinalDirection.NORTH: [304, -6],
                nb.CardinalDirection.SOUTH: [308, 6],
                nb.CardinalDirection.WEST: [300, 2],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 50),
        )

        """Road 6"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-4:s-out",
                        "I-5:n-in",
                        (s, c),
                        nb.get_weight(44, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "I-5:n-out",
                        "I-4:s-in",
                        (n, c),
                        nb.get_weight(44, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Intersection 5"""
        nb.add_intersection(
            "I-5",
            {
                nb.CardinalDirection.NORTH: [304, 50],
                nb.CardinalDirection.SOUTH: [308, 62],
                nb.CardinalDirection.EAST: [312, 54],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 50),
        )

        """Road 7"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-5:s-out",
                        "I-6:n-in",
                        (s, c),
                        nb.get_weight(38, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "I-6:n-out",
                        "I-5:s-in",
                        (n, c),
                        nb.get_weight(38, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Intersection 6"""
        nb.add_intersection(
            "I-6",
            {
                nb.CardinalDirection.NORTH: [304, 100],
                nb.CardinalDirection.SOUTH: [308, 112],
                nb.CardinalDirection.WEST: [300, 108],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 50),
        )

        """Road 11"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-6:w-out",
                        "T-3:e-in",
                        (s, c),
                        nb.get_weight(72, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "T-3:e-out",
                        "I-6:w-in",
                        (n, c),
                        nb.get_weight(72, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Turn 3"""
        nb.add_multiple_nodes(
            {
                "T-3:e-in": [228, 104],
                "T-3:e-out": [228, 108],
                "T-3:s-in": [208, 128],
                "T-3:s-out": [204, 128],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.CIRCULAR: [
                    CircularPath(
                        "T-3:e-in",
                        "T-3:s-out",
                        -90,
                        turn_radius + lane_width,
                        left_turn,
                        (s, c),
                        nb.get_weight(31, 50),
                        LaneType.CITY_ROAD,
                    ),
                    CircularPath(
                        "T-3:s-in",
                        "T-3:e-out",
                        0,
                        turn_radius,
                        right_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Road 12"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "T-3:s-out",
                        "I-8:n-in",
                        (s, c),
                        nb.get_weight(22, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "I-8:n-out",
                        "T-3:s-in",
                        (n, c),
                        nb.get_weight(22, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Intersection 8"""
        nb.add_intersection(
            "I-8",
            {
                nb.CardinalDirection.NORTH: [204, 150],
                nb.CardinalDirection.SOUTH: [208, 162],
                nb.CardinalDirection.WEST: [200, 158],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 50),
        )

        """Road 13"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-8:w-out",
                        "I-10:e-in",
                        (s, c),
                        nb.get_weight(194, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "I-10:e-out",
                        "I-8:w-in",
                        (n, c),
                        nb.get_weight(194, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Intersection 10"""
        nb.add_intersection(
            "I-10",
            {
                nb.CardinalDirection.NORTH: [-2, 150],
                nb.CardinalDirection.WEST: [-6, 158],
                nb.CardinalDirection.EAST: [6, 154],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50),
        )

        """Road 10"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-8:s-out",
                        "I-9:n-in",
                        (s, c),
                        nb.get_weight(148, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "I-9:n-out",
                        "I-8:s-in",
                        (n, c),
                        nb.get_weight(148, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Intersection 9"""
        nb.add_intersection(
            "I-9",
            {
                nb.CardinalDirection.NORTH: [204, 300],
                nb.CardinalDirection.SOUTH: [208, 312],
                nb.CardinalDirection.EAST: [212, 304],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 50),
        )

        """Road 9"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-9:e-out",
                        "I-7:w-in",
                        (s, c),
                        nb.get_weight(92, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "I-7:w-out",
                        "I-9:e-in",
                        (n, c),
                        nb.get_weight(92, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Intersection 7"""
        nb.add_intersection(
            "I-7",
            {
                nb.CardinalDirection.NORTH: [304, 300],
                nb.CardinalDirection.WEST: [300, 308],
                nb.CardinalDirection.EAST: [312, 304],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50),
        )

        """Road 8"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-6:s-out",
                        "I-7:n-in",
                        (s, c),
                        nb.get_weight(188, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "I-7:n-out",
                        "I-6:s-in",
                        (n, c),
                        nb.get_weight(188, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Road 14"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-10:w-out",
                        "T-4:e-in",
                        (s, c),
                        nb.get_weight(294, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "T-4:e-out",
                        "I-10:w-in",
                        (n, c),
                        nb.get_weight(294, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Turn 4"""
        nb.add_multiple_nodes(
            {
                "T-4:e-in": [-20, 154],
                "T-4:e-out": [-20, 158],
                "T-4:n-in": [-44, 134],
                "T-4:n-out": [-40, 134],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.CIRCULAR: [
                    CircularPath(
                        "T-4:e-in",
                        "T-4:n-out",
                        -90,
                        turn_radius,
                        right_turn,
                        (s, c),
                        nb.get_weight(31, 50),
                        LaneType.CITY_ROAD,
                    ),
                    CircularPath(
                        "T-4:n-in",
                        "T-4:e-out",
                        180,
                        turn_radius + lane_width,
                        left_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Road 15"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "T-4:n-out",
                        "T-5:s-in",
                        (s, c),
                        nb.get_weight(26, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "T-5:s-out",
                        "T-4:n-in",
                        (n, c),
                        nb.get_weight(26, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Turn 5"""
        nb.add_multiple_nodes(
            {
                "T-5:w-in": [-60, 88],
                "T-5:w-out": [-60, 84],
                "T-5:s-in": [-40, 108],
                "T-5:s-out": [-44, 108],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.CIRCULAR: [
                    CircularPath(
                        "T-5:w-in",
                        "T-5:s-out",
                        90,
                        turn_radius,
                        right_turn,
                        (s, c),
                        nb.get_weight(31, 50),
                        LaneType.CITY_ROAD,
                    ),
                    CircularPath(
                        "T-5:s-in",
                        "T-5:w-out",
                        0,
                        turn_radius + lane_width,
                        left_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Road 16"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-11:e-out",
                        "T-5:w-in",
                        (s, c),
                        nb.get_weight(54, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "T-5:w-out",
                        "I-11:e-in",
                        (n, c),
                        nb.get_weight(54, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Intersection 11"""
        nb.add_intersection(
            "I-11",
            {
                nb.CardinalDirection.NORTH: [-124, 80],
                nb.CardinalDirection.WEST: [-128, 88],
                nb.CardinalDirection.EAST: [-116, 84],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50),
        )

        """Road 17"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-12:s-out",
                        "I-11:n-in",
                        (s, c),
                        nb.get_weight(74, 50),
                        LaneType.CITY_ROAD,
                    ),
                    StraightPath(
                        "I-11:n-out",
                        "I-12:s-in",
                        (n, c),
                        nb.get_weight(74, 50),
                        LaneType.CITY_ROAD,
                    ),
                ]
            }
        )

        """Intersection 13"""
        nb.add_intersection(
            "I-13",
            {
                nb.CardinalDirection.NORTH: [-408, -212],
                nb.CardinalDirection.SOUTH: [-404, -200],
                nb.CardinalDirection.EAST: [-400, -208],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 130),
        )

        """Road 20"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "R-1:w-out",
                        "I-13:e-in",
                        (s, c),
                        nb.get_weight(52, 80),
                        LaneType.ROAD,
                    ),
                    StraightPath(
                        "I-13:e-out",
                        "R-1:w-in",
                        (n, c),
                        nb.get_weight(52, 80),
                        LaneType.ROAD,
                    ),
                ]
            }
        )

        """Intersection 14"""
        nb.add_intersection(
            "I-14",
            {
                nb.CardinalDirection.NORTH: [-408, 80],
                nb.CardinalDirection.SOUTH: [-404, 92],
                nb.CardinalDirection.EAST: [-400, 84],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 130),
        )

        """Highway 1"""
        nb.add_multiple_nodes(
            {
                "H-1:s-out:1": [-412, 70],
                "H-1:s-out:2": [-408, 70],
                "H-1:s-in:2": [-404, 70],
                "H-1:s-in:1": [-400, 70],
                "H-1:n-in:1": [-412, -190],
                "H-1:n-in:2": [-408, -190],
                "H-1:n-out:2": [-404, -190],
                "H-1:n-out:1": [-400, -190],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-13:s-out",
                        "H-1:n-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-13:s-out",
                        "H-1:n-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-1:n-out:1",
                        "I-13:s-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-1:n-out:2",
                        "I-13:s-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-14:n-out",
                        "H-1:s-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-14:n-out",
                        "H-1:s-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-1:s-out:1",
                        "I-14:n-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-1:s-out:2",
                        "I-14:n-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-1:n-in:1",
                        "H-1:s-out:1",
                        (n, c),
                        nb.get_weight(260, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-1:n-in:2",
                        "H-1:s-out:2",
                        (c, s),
                        nb.get_weight(260, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-1:s-in:2",
                        "H-1:n-out:2",
                        (c, s),
                        nb.get_weight(260, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-1:s-in:1",
                        "H-1:n-out:1",
                        (n, c),
                        nb.get_weight(260, 130),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Road 21"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-11:w-out",
                        "I-14:e-in",
                        (s, c),
                        nb.get_weight(72, 80),
                        LaneType.ROAD,
                    ),
                    StraightPath(
                        "I-14:e-out",
                        "I-11:w-in",
                        (n, c),
                        nb.get_weight(72, 80),
                        LaneType.ROAD,
                    ),
                ]
            }
        )

        """Highway 2"""
        nb.add_multiple_nodes(
            {
                "H-2:n-in:1": [-412, 102],
                "H-2:n-in:2": [-408, 102],
                "H-2:n-out:2": [-404, 102],
                "H-2:n-out:1": [-400, 102],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-14:s-out",
                        "H-2:n-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-14:s-out",
                        "H-2:n-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-2:n-out:1",
                        "I-14:s-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-2:n-out:2",
                        "I-14:s-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-2:n-in:1",
                        "T-6:n-in:1",
                        (s, c),
                        nb.get_weight(270, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-2:n-in:2",
                        "T-6:n-in:2",
                        (c, n),
                        nb.get_weight(270, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-6:n-out:2",
                        "H-2:n-out:2",
                        (c, n),
                        nb.get_weight(270, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-6:n-out:1",
                        "H-2:n-out:1",
                        (s, c),
                        nb.get_weight(270, 130),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Turn 6"""
        nb.add_multiple_nodes(
            {
                "T-6:n-in:1": [-412, 372],
                "T-6:n-in:2": [-408, 372],
                "T-6:n-out:2": [-404, 372],
                "T-6:n-out:1": [-400, 372],
                "T-6:e-in:1": [-380, 392],
                "T-6:e-in:2": [-380, 396],
                "T-6:e-out:2": [-380, 400],
                "T-6:e-out:1": [-380, 404],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.CIRCULAR: [
                    CircularPath(
                        "T-6:n-in:1",
                        "T-6:e-out:1",
                        180,
                        turn_radius + (3 * lane_width),
                        left_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                    CircularPath(
                        "T-6:n-in:2",
                        "T-6:e-out:2",
                        180,
                        turn_radius + (2 * lane_width),
                        left_turn,
                        (c, s),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                    CircularPath(
                        "T-6:e-in:2",
                        "T-6:n-out:2",
                        -90,
                        turn_radius + (1 * lane_width),
                        right_turn,
                        (c, s),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                    CircularPath(
                        "T-6:e-in:1",
                        "T-6:n-out:1",
                        -90,
                        turn_radius + (0 * lane_width),
                        right_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Highway 3"""
        nb.add_multiple_nodes(
            {
                "H-3:e-in:1": [190, 392],
                "H-3:e-in:2": [190, 396],
                "H-3:e-out:2": [190, 400],
                "H-3:e-out:1": [190, 404],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "T-6:e-out:1",
                        "H-3:e-out:1",
                        (n, c),
                        nb.get_weight(574, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-6:e-out:2",
                        "H-3:e-out:2",
                        (c, s),
                        nb.get_weight(574, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-3:e-in:2",
                        "T-6:e-in:2",
                        (c, s),
                        nb.get_weight(574, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-3:e-in:1",
                        "T-6:e-in:1",
                        (n, c),
                        nb.get_weight(574, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-15:w-out",
                        "H-3:e-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-15:w-out",
                        "H-3:e-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-3:e-out:1",
                        "I-15:w-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-3:e-out:2",
                        "I-15:w-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Road 22"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-9:s-out",
                        "I-15:n-in",
                        (s, c),
                        nb.get_weight(80, 80),
                        LaneType.ROAD,
                    ),
                    StraightPath(
                        "I-15:n-out",
                        "I-9:s-in",
                        (n, c),
                        nb.get_weight(80, 80),
                        LaneType.ROAD,
                    ),
                ]
            }
        )

        """Intersection 15"""
        nb.add_intersection(
            "I-15",
            {
                nb.CardinalDirection.NORTH: [204, 392],
                nb.CardinalDirection.WEST: [200, 400],
                nb.CardinalDirection.EAST: [212, 396],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 130),
        )

        """Highway 4"""
        nb.add_multiple_nodes(
            {
                "H-4:w-out:1": [222, 392],
                "H-4:w-out:2": [222, 396],
                "H-4:w-in:2": [222, 400],
                "H-4:w-in:1": [222, 404],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-15:e-out",
                        "H-4:w-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-15:e-out",
                        "H-4:w-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-4:w-out:1",
                        "I-15:e-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-4:w-out:2",
                        "I-15:e-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-4:w-in:1",
                        "T-7:w-in:1",
                        (n, c),
                        nb.get_weight(148, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-4:w-in:2",
                        "T-7:w-in:2",
                        (c, s),
                        nb.get_weight(148, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-7:w-out:2",
                        "H-4:w-out:2",
                        (c, s),
                        nb.get_weight(148, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-7:w-out:1",
                        "H-4:w-out:1",
                        (n, c),
                        nb.get_weight(148, 130),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Turn 7"""
        nb.add_multiple_nodes(
            {
                "T-7:w-out:1": [350, 392],
                "T-7:w-out:2": [350, 396],
                "T-7:w-in:2": [350, 400],
                "T-7:w-in:1": [350, 404],
                "T-7:n-in:1": [370, 372],
                "T-7:n-in:2": [374, 372],
                "T-7:n-out:2": [378, 372],
                "T-7:n-out:1": [382, 372],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.CIRCULAR: [
                    CircularPath(
                        "T-7:w-in:1",
                        "T-7:n-out:1",
                        90,
                        turn_radius + (3 * lane_width),
                        left_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                    CircularPath(
                        "T-7:w-in:2",
                        "T-7:n-out:2",
                        90,
                        turn_radius + (2 * lane_width),
                        left_turn,
                        (c, s),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                    CircularPath(
                        "T-7:n-in:2",
                        "T-7:w-out:2",
                        180,
                        turn_radius + (1 * lane_width),
                        right_turn,
                        (c, s),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                    CircularPath(
                        "T-7:n-in:1",
                        "T-7:w-out:1",
                        180,
                        turn_radius + (0 * lane_width),
                        right_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Highway 5"""
        nb.add_multiple_nodes(
            {
                "H-5:n-in:1": [370, 322],
                "H-5:n-in:2": [374, 322],
                "H-5:n-out:2": [378, 322],
                "H-5:n-out:1": [382, 322],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-16:s-out",
                        "H-5:n-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-16:s-out",
                        "H-5:n-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-5:n-out:1",
                        "I-16:s-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-5:n-out:2",
                        "I-16:s-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-7:n-out:1",
                        "H-5:n-out:1",
                        (n, c),
                        nb.get_weight(574, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-7:n-out:2",
                        "H-5:n-out:2",
                        (c, s),
                        nb.get_weight(574, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-5:n-in:2",
                        "T-7:n-in:2",
                        (c, s),
                        nb.get_weight(574, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-5:n-in:1",
                        "T-7:n-in:1",
                        (n, c),
                        nb.get_weight(574, 130),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Road 23"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-7:e-out",
                        "I-16:w-in",
                        (s, c),
                        nb.get_weight(58, 80),
                        LaneType.ROAD,
                    ),
                    StraightPath(
                        "I-16:w-out",
                        "I-7:e-in",
                        (n, c),
                        nb.get_weight(58, 80),
                        LaneType.ROAD,
                    ),
                ]
            }
        )

        """Intersection 16"""
        nb.add_intersection(
            "I-16",
            {
                nb.CardinalDirection.NORTH: [374, 300],
                nb.CardinalDirection.SOUTH: [378, 312],
                nb.CardinalDirection.WEST: [370, 308],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 130),
        )

        """Highway 6"""
        nb.add_multiple_nodes(
            {
                "H-6:n-in:1": [370, 72],
                "H-6:n-in:2": [374, 72],
                "H-6:n-out:2": [378, 72],
                "H-6:n-out:1": [382, 72],
                "H-6:s-out:1": [370, 290],
                "H-6:s-out:2": [374, 290],
                "H-6:s-in:2": [378, 290],
                "H-6:s-in:1": [382, 290],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-16:n-out",
                        "H-6:s-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-16:n-out",
                        "H-6:s-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-6:s-out:1",
                        "I-16:n-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-6:s-out:2",
                        "I-16:n-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-6:n-out:1",
                        "I-17:s-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-6:n-out:2",
                        "I-17:s-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-17:s-out",
                        "H-6:n-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-17:s-out",
                        "H-6:n-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-6:s-in:1",
                        "H-6:n-out:1",
                        (n, c),
                        nb.get_weight(218, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-6:s-in:2",
                        "H-6:n-out:2",
                        (c, s),
                        nb.get_weight(218, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-6:n-in:2",
                        "H-6:s-out:2",
                        (c, s),
                        nb.get_weight(218, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-6:n-in:1",
                        "H-6:s-out:1",
                        (n, c),
                        nb.get_weight(218, 130),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Road 24"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-5:e-out",
                        "I-17:w-in",
                        (s, c),
                        nb.get_weight(58, 80),
                        LaneType.ROAD,
                    ),
                    StraightPath(
                        "I-17:w-out",
                        "I-5:e-in",
                        (n, c),
                        nb.get_weight(58, 80),
                        LaneType.ROAD,
                    ),
                ]
            }
        )

        """Intersection 17"""
        nb.add_intersection(
            "I-17",
            {
                nb.CardinalDirection.NORTH: [374, 50],
                nb.CardinalDirection.SOUTH: [378, 62],
                nb.CardinalDirection.WEST: [370, 58],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 130),
        )

        """Highway 7"""
        nb.add_multiple_nodes(
            {
                "H-7:n-in:1": [370, -62],
                "H-7:n-in:2": [374, -62],
                "H-7:n-out:2": [378, -62],
                "H-7:n-out:1": [382, -62],
                "H-7:s-out:1": [370, 40],
                "H-7:s-out:2": [374, 40],
                "H-7:s-in:2": [378, 40],
                "H-7:s-in:1": [382, 40],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-17:n-out",
                        "H-7:s-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-17:n-out",
                        "H-7:s-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-7:s-out:1",
                        "I-17:n-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-7:s-out:2",
                        "I-17:n-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-7:n-out:1",
                        "I-18:s-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-7:n-out:2",
                        "I-18:s-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-18:s-out",
                        "H-7:n-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-18:s-out",
                        "H-7:n-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-7:s-in:1",
                        "H-7:n-out:1",
                        (n, c),
                        nb.get_weight(218, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-7:s-in:2",
                        "H-7:n-out:2",
                        (c, s),
                        nb.get_weight(218, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-7:n-in:2",
                        "H-7:s-out:2",
                        (c, s),
                        nb.get_weight(218, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-7:n-in:1",
                        "H-7:s-out:1",
                        (n, c),
                        nb.get_weight(218, 130),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Road 25"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-3:e-out",
                        "I-18:w-in",
                        (s, c),
                        nb.get_weight(58, 80),
                        LaneType.ROAD,
                    ),
                    StraightPath(
                        "I-18:w-out",
                        "I-3:e-in",
                        (n, c),
                        nb.get_weight(58, 80),
                        LaneType.ROAD,
                    ),
                ]
            }
        )

        """Intersection 18"""
        nb.add_intersection(
            "I-18",
            {
                nb.CardinalDirection.NORTH: [374, -84],
                nb.CardinalDirection.SOUTH: [378, -72],
                nb.CardinalDirection.WEST: [370, -76],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 130),
        )

        """Highway 8"""
        nb.add_multiple_nodes(
            {
                "H-8:s-out:1": [370, -94],
                "H-8:s-out:2": [374, -94],
                "H-8:s-in:2": [378, -94],
                "H-8:s-in:1": [382, -94],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-18:n-out",
                        "H-8:s-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-18:n-out",
                        "H-8:s-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-8:s-out:1",
                        "I-18:n-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-8:s-out:2",
                        "I-18:n-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-8:s-in:1",
                        "T-8:s-in:1",
                        (n, c),
                        nb.get_weight(236, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-8:s-in:2",
                        "T-8:s-in:2",
                        (c, s),
                        nb.get_weight(236, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-8:s-out:2",
                        "H-8:s-out:2",
                        (c, s),
                        nb.get_weight(236, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-8:s-out:1",
                        "H-8:s-out:1",
                        (n, c),
                        nb.get_weight(236, 130),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Turn 8"""
        nb.add_multiple_nodes(
            {
                "T-8:s-out:1": [370, -330],
                "T-8:s-out:2": [374, -330],
                "T-8:s-in:2": [378, -330],
                "T-8:s-in:1": [382, -330],
                "T-8:w-out:1": [350, -362],
                "T-8:w-out:2": [350, -358],
                "T-8:w-in:2": [350, -354],
                "T-8:w-in:1": [350, -350],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.CIRCULAR: [
                    CircularPath(
                        "T-8:s-in:1",
                        "T-8:w-out:1",
                        0,
                        turn_radius + (3 * lane_width),
                        left_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                    CircularPath(
                        "T-8:s-in:2",
                        "T-8:w-out:2",
                        0,
                        turn_radius + (2 * lane_width),
                        left_turn,
                        (c, s),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                    CircularPath(
                        "T-8:w-in:2",
                        "T-8:s-out:2",
                        90,
                        turn_radius + (1 * lane_width),
                        right_turn,
                        (c, s),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                    CircularPath(
                        "T-8:w-in:1",
                        "T-8:s-out:1",
                        90,
                        turn_radius + (0 * lane_width),
                        right_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Highway 9"""
        nb.add_multiple_nodes(
            {
                "H-9:w-out:1": [72, -362],
                "H-9:w-out:2": [72, -358],
                "H-9:w-in:2": [72, -354],
                "H-9:w-in:1": [72, -350],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-19:e-out",
                        "H-9:w-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-19:e-out",
                        "H-9:w-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-9:w-out:1",
                        "I-19:e-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-9:w-out:2",
                        "I-19:e-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-8:w-out:1",
                        "H-9:w-out:1",
                        (n, c),
                        nb.get_weight(278, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-8:w-out:2",
                        "H-9:w-out:2",
                        (c, s),
                        nb.get_weight(278, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-9:w-in:2",
                        "T-8:w-in:2",
                        (c, s),
                        nb.get_weight(278, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-9:w-in:1",
                        "T-8:w-in:1",
                        (n, c),
                        nb.get_weight(278, 130),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Road 26"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-2:n-out",
                        "I-19:s-in",
                        (s, c),
                        nb.get_weight(148, 80),
                        LaneType.ROAD,
                    ),
                    StraightPath(
                        "I-19:s-out",
                        "I-2:n-in",
                        (n, c),
                        nb.get_weight(148, 80),
                        LaneType.ROAD,
                    ),
                ]
            }
        )

        """Intersection 19"""
        nb.add_intersection(
            "I-19",
            {
                nb.CardinalDirection.SOUTH: [58, -350],
                nb.CardinalDirection.WEST: [50, -354],
                nb.CardinalDirection.EAST: [62, -358],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 130),
        )

        """Highway 10"""
        nb.add_multiple_nodes(
            {
                "H-10:w-out:1": [16, -362],
                "H-10:w-out:2": [16, -358],
                "H-10:w-in:2": [16, -354],
                "H-10:w-in:1": [16, -350],
                "H-10:e-in:1": [40, -362],
                "H-10:e-in:2": [40, -358],
                "H-10:e-out:2": [40, -354],
                "H-10:e-out:1": [40, -350],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-19:w-out",
                        "H-10:e-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-19:w-out",
                        "H-10:e-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-10:e-out:1",
                        "I-19:w-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-10:e-out:2",
                        "I-19:w-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-20:e-out",
                        "H-10:w-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-20:e-out",
                        "H-10:w-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-10:w-out:1",
                        "I-20:e-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-10:w-out:2",
                        "I-20:e-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-10:e-in:1",
                        "H-10:w-out:1",
                        (n, c),
                        nb.get_weight(14, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-10:e-in:2",
                        "H-10:w-out:2",
                        (c, s),
                        nb.get_weight(14, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-10:w-in:2",
                        "H-10:e-out:2",
                        (c, s),
                        nb.get_weight(14, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-10:w-in:1",
                        "H-10:e-out:1",
                        (n, c),
                        nb.get_weight(14, 130),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Road 26"""
        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-1:n-out",
                        "I-20:s-in",
                        (s, c),
                        nb.get_weight(148, 80),
                        LaneType.ROAD,
                    ),
                    StraightPath(
                        "I-20:s-out",
                        "I-1:n-in",
                        (n, c),
                        nb.get_weight(148, 80),
                        LaneType.ROAD,
                    ),
                ]
            }
        )

        """Intersection 20"""
        nb.add_intersection(
            "I-20",
            {
                nb.CardinalDirection.SOUTH: [2, -350],
                nb.CardinalDirection.WEST: [-6, -354],
                nb.CardinalDirection.EAST: [6, -358],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 130),
        )

        """Highway 11"""
        nb.add_multiple_nodes(
            {
                "H-11:e-in:1": [-16, -362],
                "H-11:e-in:2": [-16, -358],
                "H-11:e-out:2": [-16, -354],
                "H-11:e-out:1": [-16, -350],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-20:w-out",
                        "H-11:e-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-20:w-out",
                        "H-11:e-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-11:e-out:1",
                        "I-20:w-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-11:e-out:2",
                        "I-20:w-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-11:e-in:1",
                        "T-9:w-in:1",
                        (n, c),
                        nb.get_weight(364, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-11:e-in:2",
                        "T-9:w-in:2",
                        (c, s),
                        nb.get_weight(364, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-9:w-out:2",
                        "H-11:e-out:2",
                        (c, s),
                        nb.get_weight(364, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-9:w-out:1",
                        "H-11:e-out:1",
                        (n, c),
                        nb.get_weight(364, 130),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Turn 9"""
        nb.add_multiple_nodes(
            {
                "T-9:w-in:1": [-380, -362],
                "T-9:w-in:2": [-380, -358],
                "T-9:w-out:2": [-380, -354],
                "T-9:w-out:1": [-380, -350],
                "T-9:s-out:1": [-412, -330],
                "T-9:s-out:2": [-408, -330],
                "T-9:s-in:2": [-404, -330],
                "T-9:s-in:1": [-400, -330],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.CIRCULAR: [
                    CircularPath(
                        "T-9:w-in:1",
                        "T-9:s-out:1",
                        -90,
                        turn_radius + (3 * lane_width),
                        left_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                    CircularPath(
                        "T-9:w-in:2",
                        "T-9:s-out:2",
                        -90,
                        turn_radius + (2 * lane_width),
                        left_turn,
                        (c, s),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                    CircularPath(
                        "T-9:s-in:2",
                        "T-9:w-out:2",
                        0,
                        turn_radius + (1 * lane_width),
                        right_turn,
                        (c, s),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                    CircularPath(
                        "T-9:s-in:1",
                        "T-9:w-out:1",
                        0,
                        turn_radius + (0 * lane_width),
                        right_turn,
                        (n, c),
                        nb.get_weight(31, 50),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        """Highway 12"""
        nb.add_multiple_nodes(
            {
                "H-12:s-out:1": [-412, -222],
                "H-12:s-out:2": [-408, -222],
                "H-12:s-in:2": [-404, -222],
                "H-12:s-in:1": [-400, -222],
            }
        )

        nb.add_multiple_paths(
            {
                nb.PathType.STRAIGHT: [
                    StraightPath(
                        "I-13:n-out",
                        "H-12:s-in:1",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "I-13:n-out",
                        "H-12:s-in:2",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-12:s-out:1",
                        "I-13:n-in",
                        (n, c),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-12:s-out:2",
                        "I-13:n-in",
                        (c, s),
                        nb.get_weight(10, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-9:s-out:1",
                        "H-12:s-out:1",
                        (n, c),
                        nb.get_weight(108, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "T-9:s-out:2",
                        "H-12:s-out:2",
                        (c, s),
                        nb.get_weight(108, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-12:s-in:2",
                        "T-9:s-in:2",
                        (c, s),
                        nb.get_weight(108, 130),
                        LaneType.HIGHWAY,
                    ),
                    StraightPath(
                        "H-12:s-in:1",
                        "T-9:s-in:1",
                        (n, c),
                        nb.get_weight(108, 130),
                        LaneType.HIGHWAY,
                    ),
                ]
            }
        )

        nb.build_paths(net)

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        self.road = road
        return net

    def _get_shortest_path(
        self, startpoint: tuple[str, str, int], destination: tuple[str, str, int]
    ) -> list[tuple[str, str, int]]:
        shortest_path = [startpoint[0]] + self.shortest_paths[startpoint[1]][
            destination[1]
        ]
        return [
            (shortest_path[i], shortest_path[i + 1], None)
            for i in range(len(shortest_path) - 1)
        ]

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        entry_edge = self._get_random_edge()
        exit_edge = self._get_random_destination_different_from(entry_edge)

        while entry_edge[0] in exit_edge or entry_edge[1] in exit_edge:
            logger.info(
                f"\t_spawn_vehicle                         :: Element in 'entry_edge' was in 'exit_edge' -- {entry_edge} ~> {exit_edge}"
            )
            exit_edge = self._get_random_edge()

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        speed = self.road.network.get_lane(entry_edge).speed_limit if not None else 25
        # TODO: Handle speed in a better way
        vehicle = vehicle_type.make_on_lane(
            self.road,
            entry_edge,
            longitudinal=(
                longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            speed=speed + self.np_random.normal() * speed_deviation,
        )
        # Not adding the vehicle, if it is too close to another vehicle
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return

        routes_logger.info(
            f"\t_spawn_vehicele :: planning route {entry_edge} ~> {exit_edge}"
        )
        vehicle.route = self._get_shortest_path(entry_edge, exit_edge)

        vehicle.check_collisions = False
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _categorize_edges_by_type(self):
        """
        Categorize edges based on the end vertex type.
         - Intersection edges: edges whose end vertex name starts with 'I'
         - Roundabout edges: edges whose end vertex name starts with 'R'
         - Highway edges: edges whose end vertex name starts with 'H'
         - Turn edges: edges whose end vertex name starts with 'T'
        """
        self.has_been_categorized = True
        self.I_edges = []
        self.R_edges = []
        self.H_edges = []
        self.T_edges = []

        for edge in list(self.local_graph_net.edges):

            # Skip the edges we cannot get back
            edge_lane = self.road.network.get_lane(edge)
            close_edge = self.road.network.get_closest_lane_index(
                edge_lane.position(0, 0), edge_lane.heading_at(60)
            )
            if edge != close_edge:
                logger.info(
                    f"\t_categorize_edges_by_type          :: close_edge != edge -- {close_edge} : {edge}"
                )
                self.local_graph_net.remove_edge(*edge)
                continue

            end_node = edge[1]
            if end_node.startswith("I-"):
                self.I_edges.append(edge)
            elif end_node.startswith("R-"):
                self.R_edges.append(edge)
            elif end_node.startswith("H-"):
                self.H_edges.append(edge)
            elif end_node.startswith("T-"):
                self.T_edges.append(edge)

    def _get_balanced_random_edge(self) -> tuple[str, str, int]:
        categories: list[list[tuple[str, str, int]]] = [
            self.I_edges,
            self.R_edges,
            self.H_edges,
            self.T_edges,
        ]
        category_index: int = self.episode_count % len(categories)
        chosen_category: list[tuple[str, str, int]] = categories[category_index]

        edge = self.get_random_edge_from(chosen_category)

        # Validate the edge can be found again
        edge_lane = self.road.network.get_lane(edge)
        close_edge = self.road.network.get_closest_lane_index(
            edge_lane.position(0, 0), edge_lane.heading_at(60)
        )

        while edge != close_edge:
            logger.info(
                f"\t_get_balanced_random_edge              :: close_edge != edge -- {close_edge} : {edge}"
            )
            edge = self.get_random_edge_from(chosen_category)
            edge_lane = self.road.network.get_lane(edge)
            close_edge = self.road.network.get_closest_lane_index(
                edge_lane.position(0, 0), edge_lane.heading_at(60)
            )

        return edge

    def _get_random_edge(self) -> tuple[str, str, int]:
        edges = list(self.local_graph_net.edges)
        edge = self.get_random_edge_from(edges)

        # Validate the edge can be found again
        edge_lane = self.road.network.get_lane(edge)
        close_edge = self.road.network.get_closest_lane_index(
            edge_lane.position(0, 0), edge_lane.heading_at(60)
        )

        while edge != close_edge:
            logger.info(
                f"\t_get_random_edge                       :: close_edge != edge -- {close_edge} : {edge}"
            )
            self.local_graph_net.remove_edge(*edge)

            edge = self.get_random_edge_from(edges)
            edge_lane = self.road.network.get_lane(edge)
            close_edge = self.road.network.get_closest_lane_index(
                edge_lane.position(0, 0), edge_lane.heading_at(60)
            )

        return edge

    def _get_random_destination_different_from(
        self, start_edge
    ) -> tuple[str, str, int]:
        destination = self._get_random_edge()

        # Validate that no vertex from 'start_edge' is in 'destination'
        while start_edge[0] in destination or start_edge[1] in destination:
            logger.info(
                f"\t_get_random_destination_different_from :: Element in 'startpoint' was in 'destination' -- {start_edge} ~> {destination}"
            )
            destination = self._get_random_edge()

        return destination

    def get_destination_close_to(
        self, start_edge, ego_destination
    ) -> tuple[str, str, int]:
        for _ in range(10):
            candidate = self._get_random_destination_different_from(start_edge)
            if ego_destination in candidate:
                return candidate

        return None

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # print("\n---::: New make vechicle :::---")
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            startpoint = self._get_balanced_random_edge()
            destination = self._get_random_destination_different_from(startpoint)

            ego_lane = self.road.network.get_lane(startpoint)
            ego_longtitudinal = min(50, ego_lane.length - 10)
            ego_pos = ego_lane.position(ego_longtitudinal, 0)
            ego_heading = ego_lane.heading_at(ego_longtitudinal)

            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_pos,
                speed=10,
                heading=ego_heading,
            )

            try:
                routes_logger.info(
                    f"\t_make_vehicele  :: ego vehicle planning route {startpoint} ~> {destination}"
                )
                # print(f"destination[1]: {destination[1]} :: plan_route_to({destination[1]})")

                ego_vehicle.route = self._get_shortest_path(startpoint, destination)
                # ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                # ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                print("Got an attribute error")
                logger.warning(
                    f"In episode '{self.episode_count}': AttributeError while planning ego route"
                )
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)

        # ---::: Code to spawn other vehicles goes here :::---

        # Use the ego vehicle's route. This should be a list of edges like [(start_node, end_node, lane_index), ...]
        candidate_edges = list(ego_vehicle.route)

        # For each edge in the route, add all side lanes
        for edge in ego_vehicle.route:
            side_lane_indexes = self.road.network.all_side_lanes(edge)
            for sl_idx in side_lane_indexes:
                candidate_edges.append(sl_idx)

        # Remove duplicates if any
        candidate_edges = list(dict.fromkeys(candidate_edges))
        ego_final_node = destination
        close_to_ego_dest_probability = 0.7

        vehicle_behind = 5
        vehicles_behind_placed = 0
        behind_spacing = 10.0
        num_other_vehicles = n_vehicles

        while vehicles_behind_placed < vehicle_behind and num_other_vehicles > 0:
            behind_longitudinal = ego_longtitudinal - (
                behind_spacing * (vehicles_behind_placed + 1)
            )
            if behind_longitudinal < 0:
                break

            pos_behind = ego_lane.position(behind_longitudinal, 0)
            heading_behind = ego_lane.heading_at(behind_longitudinal)

            if (
                self.road.network.get_lane_type(startpoint[0], startpoint[1])
                == LaneType.HIGHWAY
            ):
                speed_behind = np.random.uniform(20, 30)
            else:
                speed_behind = np.random.uniform(5, 20)

            other_vehicle = vehicle_type(
                self.road, pos_behind, heading=heading_behind, speed=speed_behind
            )

            no_collision = True
            for v in self.road.vehicles:
                if np.linalg.norm(v.position - other_vehicle.position) < 10:
                    no_collision = False
                    break

            if no_collision:
                # Route for behind vehicle: also try close to ego's final node
                start_edge = startpoint
                if random.random() < close_to_ego_dest_probability:
                    vehicle_destination = self.get_destination_close_to(
                        start_edge, ego_final_node
                    )
                    if vehicle_destination is None:
                        vehicle_destination = (
                            self._get_random_destination_different_from(start_edge)
                        )
                else:
                    vehicle_destination = self._get_random_destination_different_from(
                        start_edge
                    )

                try:
                    other_vehicle.route = self._get_shortest_path(
                        start_edge, vehicle_destination
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not get shortest path for behind vehicle: {e}"
                    )
                    other_vehicle.route = []

                other_vehicle.randomize_behavior()
                self.road.vehicles.append(other_vehicle)
                vehicles_behind_placed += 1
                num_other_vehicles -= 1

            else:
                break

        # Spawn the other vehicles
        _attempts = 10
        for _ in range(num_other_vehicles):
            placed = False
            for _attempt in range(_attempts):  # up to 10 attempts to place a vehicle
                if not candidate_edges:
                    # If no candidate lanes are available, break early.
                    break

                # Pick a random edge and get its corresponding lane
                edge = random.choice(candidate_edges)
                lane = self.road.network.get_lane(edge)

                # Choose a random longitudinal position along this lane
                longitudinal = np.random.uniform(0, lane.length)
                pos = lane.position(longitudinal, 0)
                heading = lane.heading_at(longitudinal)

                if (
                    self.road.network.get_lane_type(edge[0], edge[1])
                    == LaneType.HIGHWAY
                ):
                    speed = np.random.uniform(20, 30)
                else:
                    speed = np.random.uniform(5, 20)

                other_vehicle = vehicle_type(
                    self.road, pos, heading=heading, speed=speed
                )
                other_vehicle.check_collisions = False

                # Check collision with existing vehicles
                no_collision = True
                for v in self.road.vehicles:
                    if (
                        np.linalg.norm(v.position - other_vehicle.position) < 10
                    ):  # 10 meters safe distance
                        no_collision = False
                        break

                if no_collision:
                    start_edge = edge

                    if random.random() < close_to_ego_dest_probability:
                        # Get destination close to ego_destination
                        vehicle_destination = self.get_destination_close_to(
                            start_edge, ego_final_node
                        )
                        if vehicle_destination is None:
                            vehicle_destination = (
                                self._get_random_destination_different_from(start_edge)
                            )
                    else:
                        # Get random destination
                        vehicle_destination = (
                            self._get_random_destination_different_from(start_edge)
                        )

                    try:
                        routes_logger.info(
                            f"\t_make_vehicele  :: planning route {start_edge} ~> {vehicle_destination}"
                        )
                        other_vehicle.route = self._get_shortest_path(
                            start_edge, vehicle_destination
                        )
                    except Exception as e:
                        logger.warning(
                            f"In episode '{self.episode_count}': AttributeError while planning ego route"
                        )
                        other_vehicle.route = []

                    other_vehicle.randomize_behavior()
                    self.road.vehicles.append(other_vehicle)
                    placed = True
                    break

            if not placed:
                logger.info(
                    f"Could not place a new vehicle without collision after {_attempts} attempts."
                )

        # ---::: End of code for spawning other vehicles :::---

        for v in self.road.vehicles:  # Prevent early collisions
            if (
                v is not ego_vehicle
                and np.linalg.norm(v.position - ego_vehicle.position) < 20
            ):
                self.road.vehicles.remove(v)

    # Note this reward function is just generic from another template
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        MIN_REWARD = -1
        MAX_REWARD = 3.1
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    MIN_REWARD,
                    MAX_REWARD,
                ],
                [0, 1],
            )
        if self.config["normalize_reward"]:
            reward = np.clip(reward, 0, 1)

        return reward

    # Note this reward function is just generic from another template
    def _rewards(self, action: Action) -> dict[str, float]:
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        distance = np.absolute(self.vehicle.remaining_route_nodes)
        distance_reward = 1 / (np.power(2, (distance / 2)) - 1)
        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": np.clip(
                scaled_speed, -1, 1
            ),  ## now gives negative rewards for driving too slow, but no more than 1 reward for driving fast
            "distance_from_goal": distance_reward,
            "headway_evaluation": self.vehicle.headway_evaluation,
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
        return (
            self.time >= self.config["duration"]
            or self.vehicle.remaining_route_nodes == 1
        )
