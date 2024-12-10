from __future__ import annotations

import copy
import logging
import math
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
from highway_env.road.road import RoadNetwork, WeightedRoadnetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.envs.weighted_utils import WeightedUtils

# Debug logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("stovring.log")
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Episode and route logger
routes_logger = logging.getLogger(__name__ + ".routes")
routes_logger.setLevel(logging.INFO)
file_handler_routes = logging.FileHandler("stovring_routes.log")
routes_formatter = logging.Formatter('%(message)s')
file_handler_routes.setFormatter(routes_formatter)
routes_logger.addHandler(file_handler_routes)
routes_logger.propagate = False

# Shortest path logger
shortest_path_logger = logging.getLogger(__name__ + ".routes")
shortest_path_logger.setLevel(logging.INFO)
file_handler_routes = logging.FileHandler("stovring_shortest_paths.log")
shortest_path_formatter = logging.Formatter('%(message)s')
file_handler_routes.setFormatter(shortest_path_formatter)
shortest_path_logger.addHandler(file_handler_routes)
shortest_path_logger.propagate = False

class Stovring(AbstractEnv, WeightedUtils):
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

    def calculate_shortest_paths(self):
        file_name = "stovring-paths.pkl"
        
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                routes = pickle.load(f)
        else:
            routes = {}
        
        edges = self.road.network.graph_net.edges

        total_edges = (len(edges) * (len(edges) - 1)) 
        edges_left = total_edges
        
        for _, s, _ in edges:
            if s in routes:
                edges_left -= len(routes[s])
                
            edges_left -= 1
            
        if edges_left < 0:
            edges_left = 0
            
        avg_time = None
        count = 0
        
        
        # Iterate over each location as the source
        i = 0
        for _, startpoint, _ in edges:
            # shortest_path_logger.info(f"---::: {i}/{len(edges)-1} @ Beginning on '{startpoint}' :::---")
            print(f"---::: {i}/{len(edges)} @ Beginning on '{startpoint}' :::---")

            # Create a nested dictionary for this source
            if startpoint not in routes:
                routes[startpoint] = {}

                
            # Iterate over each location as the destination
            j = 0
            for _, destination, _ in edges:
                if startpoint == destination:
                    j += 1
                    continue

                start_t = time.time()
                if destination in routes[startpoint]:
                    # shortest_path_logger.info(f"\t$$$ Skipping already calculated route from '{startpoint}' ~> '{destination}'\t @ path is: {routes[startpoint][destination]}")
                    print(f"\t$$$ Skipping already calculated route from '{startpoint}' ~> '{destination}'\t")
                        
                    j += 1
                    edges_left -= 1
                    continue
                
                
                # Compute the route from src to dst
                result = self.road.network.shortest_path(startpoint, destination)

                # Store the result in the nested dict structure
                routes[startpoint][destination] = result
                    
                # Once the dictionary is complete, write it to a JSON file
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
                print(f"\t$$$ estimate time left: {time_left_str}\t $$$ start: {i}/{len(edges)-1} @ dest.: {j}/{len(edges)-2} @ '{startpoint}' ~> '{destination}'")
                
                j += 1
            
            i +=1
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
    
        if not hasattr(self, "shortest_paths"):
            with open("carpet-city-paths.pkl", "rb") as f:
                self.shortest_paths = pickle.load(f)
        
        if not hasattr(self, "local_graph_net"):
            self.local_graph_net = copy.deepcopy(self.road.network.graph_net)
        
        if not hasattr(self, "shortest_paths"):
            with open("stovring-paths.pkl", "rb") as f:
                self.shortest_paths = pickle.load(f)
        
        if not hasattr(self, "has_been_categorized"):
            self._categorize_edges_by_type()
        
        self._make_vehicles(self.config["vehicles_count"])
        self.episode_count += 1

    def _make_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""

        net = WeightedRoadnetwork()
        nb = NetworkBuilder()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        left_turn = False
        right_turn = True
        lane_width = AbstractLane.DEFAULT_WIDTH
        small_radius = 12
        medium_radius = 20

        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [-206, 3408]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        radii = [radius, radius + 4]
        line = [[c, s], [n, c]]

        """Roundabout 6"""
        
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
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
                    speed_limit=40,
                ),
                nb.get_weight(15.7, 50),
                LaneType.ROUNDABOUT
            )


        # Access lanes: (r)oad/(s)ine
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2 * dev  # [m]

        delta_en = dev - delta_st
        w = 2 * np.pi / dev
        
        
        """Road 1"""
        # East Enter straight lane
        net.add_lane(
            "I-29:w-out", "ees",
            StraightLane(
                [0, 3406],
                [dev / 2 + center[0], -2 + center[1]],
                line_types=(s, c),
                priority=3,
                    speed_limit=40,
            ),
            nb.get_weight(164, 80),
            LaneType.ROAD
        )
        # East Enter sine lane
        net.add_lane(
            "ees", "ee",
            SineLane(
                [dev / 2 + center[0], -2 - a + center[1]],
                [dev / 2 - delta_st + center[0], -2 - a + center[1]],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
                speed_limit=40,
            ),
            nb.get_weight(18, 80),
            LaneType.ROUNDABOUT
        )
        # East Exit sine lane
        net.add_lane(
            "ex",
            "exs",
            SineLane(
                [-dev / 2 + delta_en + center[0], 2 + a + center[1]],
                [dev / 2 + center[0], 2 + a + center[1]],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
                speed_limit=40,
            ),
            nb.get_weight(18, 80),
            LaneType.ROUNDABOUT
        )
        # East Exit straight lane
        net.add_lane(
            "exs", "I-29:w-in",
            StraightLane(
                [dev / 2 + center[0], 2 + center[1]],
                [0, 3410],
                line_types=(n, c),
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(164, 80),
            LaneType.ROAD
        )


        """Road 7"""
        # North Enter straight lane
        net.add_lane(
            "R-4:s-out",
            "nes",
            StraightLane(
                [-208, 2217],
                [-2 + center[0], -dev / 2 + center[1]],
                line_types=(s, c),
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(1149, 80),
            LaneType.ROAD
        )
        # Nort Enter sine lane
        net.add_lane(
            "nes",
            "ne",
            SineLane(
                [-2 - a + center[0], -dev / 2 + center[1]],
                [-2 - a + center[0], -dev / 2 + delta_st + center[1]],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
                speed_limit=40,
            ),
            nb.get_weight(18, 80),
            LaneType.ROUNDABOUT
        )
        # Nort Exit sine lane
        net.add_lane(
            "nx",
            "nxs",
            SineLane(
                [2 + a + center[0], dev / 2 - delta_en + center[1]],
                [2 + a + center[0], -dev / 2 + center[1]],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
                speed_limit=40,
            ),
            nb.get_weight(18, 80),
            LaneType.ROUNDABOUT
        )
        # North Exit straight lane
        net.add_lane(
            "nxs", "R-4:s-in",
            StraightLane(
                [2 + center[0], -dev / 2 + center[1]],
                [-204, 2217],
                line_types=(n, c),
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(1149, 80),
            LaneType.ROAD
        )


        """Road 81"""
        # West Enter straight lane
        net.add_lane(
            "T-30:e-out", "wes",
            StraightLane(
                [-300, 3410],
                [-dev / 2 + center[0], 2 + center[1]],
                line_types=(s, c),
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(88, 80),
            LaneType.ROAD
        )
        # West Enter sine lane
        net.add_lane(
            "wes",
            "we",
            SineLane(
                [-dev / 2 + center[0], 2 + a + center[1]],
                [-dev / 2 + delta_st + center[0], 2 + a + center[1]],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
                speed_limit=40,
            ),
            nb.get_weight(18, 80),
            LaneType.ROUNDABOUT
        )
        # West Exit sine lane
        net.add_lane(
            "wx",
            "wxs",
            SineLane(
                [dev / 2 - delta_en + center[0], -2 - a + center[1]],
                [-dev / 2 + center[0], -2 - a + center[1]],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
                speed_limit=40,
            ),
            nb.get_weight(18, 80),
            LaneType.ROUNDABOUT
        )
        # West Exit straight lane
        net.add_lane(
            "wxs", "T-30:e-in",
            StraightLane(
                [-dev / 2 + center[0], -2 + center[1]],
                [-300, 3406],
                line_types=(n, c),
                priority=3,
                speed_limit=40,
            ),
            nb.get_weight(88, 80),
            LaneType.ROAD
        )
        
        
        
        """
        Nodes comming out of the roundabout
        
        north exit -> "R-4:s-in"  : [-204, 2217]
        east  exit -> "I-29:w-in"  : [0, 3410]
        west  exit -> "T-30:e-in" : [-300, 3406]
        """
        

        """Intersection 29"""
        nb.add_intersection(
            "I-29",
            {
                nb.CardinalDirection.NORTH : [4, 3402],
                nb.CardinalDirection.WEST  : [0, 3410],
                nb.CardinalDirection.EAST  : [12, 3406],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 80)
        )
        
        
        """Roundabout 4"""
        nb.add_roundabout(
            "R-4",
            {
                nb.CardinalDirection.NORTH : [-208, 2175],
                nb.CardinalDirection.SOUTH : [-204, 2217],
                nb.CardinalDirection.EAST  : [-185, 2194],
            },
            nb.get_weight(90, 80)
        )
        
        
        """Intersection 28"""
        nb.add_intersection(
            "I-28",
            {
                nb.CardinalDirection.NORTH : [4, 2954],
                nb.CardinalDirection.SOUTH : [8, 2968],
                nb.CardinalDirection.EAST  : [12, 2960],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 80)
        )
        
        
        """Road 8"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-29:n-out", "I-28:s-in", (s,c), nb.get_weight(434, 80), LaneType.ROAD),
                StraightPath("I-28:s-out", "I-29:n-in", (n,c), nb.get_weight(434, 80), LaneType.ROAD),
            ]
        })
        
        
        """Intersection 24"""
        nb.add_intersection(
            "I-24",
            {
                nb.CardinalDirection.NORTH : [4, 2752],
                nb.CardinalDirection.SOUTH : [8, 2764],
                nb.CardinalDirection.EAST  : [12, 2756],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 80)
        )
        
        
        """Road 11"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-28:n-out", "I-24:s-in", (s,c), nb.get_weight(190, 80), LaneType.ROAD),
                StraightPath("I-24:s-out", "I-28:n-in", (n,c), nb.get_weight(190, 80), LaneType.ROAD),
            ]
        })
        
        
        """Intersection 25"""
        nb.add_intersection(
            "I-25",
            {
                nb.CardinalDirection.SOUTH : [670, 2792],
                nb.CardinalDirection.WEST  : [662, 2788],
                nb.CardinalDirection.EAST  : [674, 2784],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 17"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-24:e-out", "T-19:w-in", (s,c), nb.get_weight(300, 50), LaneType.CITY_ROAD),
                StraightPath("T-19:w-out", "I-24:e-in", (n,c), nb.get_weight(300, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Turn 19 & Turn 20"""
        nb.add_multiple_nodes({
            "T-19:w-out" : [312, 2756],
            "T-19:w-in"  : [312, 2760],
            "T-19:s-out" : [324, 2772],
            "T-19:s-in"  : [328, 2772],
            "T-20:e-in"  : [340, 2784],
            "T-20:e-out" : [340, 2788],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-19:w-in", "T-19:s-out",   90, small_radius,              right_turn, (s,c), nb.get_weight(19, 50), LaneType.CITY_ROAD),
                CircularPath("T-19:s-in", "T-19:w-out",    0, small_radius + lane_width, left_turn,  (n,c), nb.get_weight(19, 50), LaneType.CITY_ROAD),
                CircularPath("T-19:s-out", "T-20:e-out", 180, small_radius + lane_width, left_turn,  (s,c), nb.get_weight(19, 50), LaneType.CITY_ROAD),
                CircularPath("T-20:e-in", "T-19:s-in",   -90, small_radius,              right_turn, (n,c), nb.get_weight(19, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Road 16"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-20:e-out", "I-25:w-in", (n,c), nb.get_weight(322, 50), LaneType.CITY_ROAD),
                StraightPath("I-25:w-out", "T-20:e-in", (s,c), nb.get_weight(322, 50), LaneType.CITY_ROAD),
            ]
        })


        """Road 10"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-28:e-out", "T-21:w-in", (s,c), nb.get_weight(632, 50), LaneType.CITY_ROAD),
                StraightPath("T-21:w-out", "I-28:e-in", (n,c), nb.get_weight(632, 50), LaneType.CITY_ROAD),
            ]
        })


        """Turn 21"""
        nb.add_multiple_nodes({
            "T-21:w-out" : [646, 2960],
            "T-21:w-in"  : [646, 2964],
            "T-21:n-in"  : [666, 2940],
            "T-21:n-out" : [670, 2940],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-21:w-in", "T-21:n-out",  90, medium_radius + lane_width, left_turn,  (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-21:n-in", "T-21:w-out", 180, medium_radius,              right_turn, (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Road 15"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-21:n-out", "I-25:s-in", (n,c), nb.get_weight(148, 50), LaneType.CITY_ROAD),
                StraightPath("I-25:s-out", "T-21:n-in", (s,c), nb.get_weight(148, 50), LaneType.CITY_ROAD),
            ]
        })


        """Road 2"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-29:e-out", "T-24:w-in", (s,c), nb.get_weight(632, 80), LaneType.ROAD),
                StraightPath("T-24:w-out", "I-29:e-in", (n,c), nb.get_weight(632, 80), LaneType.ROAD),
            ]
        })


        """Turn 24"""
        nb.add_multiple_nodes({
            "T-24:w-out" : [462, 3406],
            "T-24:w-in"  : [462, 3410],
            "T-24:n-in"  : [482, 3386],
            "T-24:n-out" : [486, 3386],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-24:w-in", "T-24:n-out",  90, medium_radius + lane_width, left_turn,  (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-24:n-in", "T-24:w-out", 180, medium_radius,              right_turn, (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 3"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-24:n-out", "T-22:s-in", (s,c), nb.get_weight(286, 50), LaneType.CITY_ROAD),
                StraightPath("T-22:s-out", "T-24:n-in", (n,c), nb.get_weight(286, 50), LaneType.CITY_ROAD),
            ]
        })


        """Turn 22"""
        nb.add_multiple_nodes({
            "T-22:s-out" : [482, 3100],
            "T-22:s-in"  : [486, 3100],
            "T-22:e-in"  : [506, 3076],
            "T-22:e-out" : [506, 3080],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-22:s-in", "T-22:e-out",   0, medium_radius,              right_turn, (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-22:e-in", "T-22:s-out", -90, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 4"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-22:e-out", "T-23:w-in", (s,c), nb.get_weight(356, 50), LaneType.CITY_ROAD),
                StraightPath("T-23:w-out", "T-22:e-in", (n,c), nb.get_weight(356, 50), LaneType.CITY_ROAD),
            ]
        })


        """Turn 23"""
        nb.add_multiple_nodes({
            "T-23:w-out" : [868, 3076],
            "T-23:w-in"  : [868, 3080],
            "T-23:s-in"  : [892, 3100],
            "T-23:s-out" : [888, 3100],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-23:w-in", "T-23:s-out",  90, medium_radius,              right_turn, (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-23:s-in", "T-23:w-out",   0, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })


        """Road 5"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-23:s-out", "T-25:n-in", (s,c), nb.get_weight(286, 50), LaneType.CITY_ROAD),
                StraightPath("T-25:n-out", "T-23:s-in", (n,c), nb.get_weight(286, 50), LaneType.CITY_ROAD),
            ]
        })


        """Turn 25"""
        nb.add_multiple_nodes({
            "T-25:n-out" : [892, 3386],
            "T-25:n-in"  : [888, 3386],
            "T-25:e-in"  : [912, 3406],
            "T-25:e-out" : [912, 3410],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-25:n-in", "T-25:e-out", 180, medium_radius + lane_width, left_turn,  (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-25:e-in", "T-25:n-out", -90, medium_radius,              right_turn, (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })


        """Road 6"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-25:e-out", "T-26:w-in", (s,c), nb.get_weight(880, 50), LaneType.CITY_ROAD),
                StraightPath("T-26:w-out", "T-25:e-in", (n,c), nb.get_weight(880, 50), LaneType.CITY_ROAD),
            ]
        })


        """Turn 26"""
        nb.add_multiple_nodes({
            "T-26:w-out" : [1796, 3406],
            "T-26:w-in"  : [1796, 3410],
            "T-26:n-in"  : [1816, 3386],
            "T-26:n-out" : [1820, 3386],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-26:w-in", "T-26:n-out",  90, medium_radius + lane_width, left_turn,  (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-26:n-in", "T-26:w-out", 180, medium_radius,              right_turn, (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Road 14"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-25:e-out", "I-26:w-in", (s,c), nb.get_weight(100, 50), LaneType.CITY_ROAD),
                StraightPath("I-26:w-out", "I-25:e-in", (n,c), nb.get_weight(100, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 26"""
        nb.add_intersection(
            "I-26",
            {
                nb.CardinalDirection.NORTH : [778, 2780],
                nb.CardinalDirection.WEST  : [774, 2788],
                nb.CardinalDirection.EAST  : [786, 2784],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 13"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-26:e-out", "R-5:w-in", (s,c), nb.get_weight(285, 50), LaneType.CITY_ROAD),
                StraightPath("R-5:w-out", "I-26:e-in", (n,c), nb.get_weight(285, 50), LaneType.CITY_ROAD),
            ]
        })

        
        """Roundabout 5"""
        nb.add_roundabout(
            "R-5",
            {
                nb.CardinalDirection.NORTH : [1090, 2765],
                nb.CardinalDirection.WEST  : [1071, 2788],
                nb.CardinalDirection.EAST  : [1113, 2784],
            },
            nb.get_weight(90, 50)
        )
        
        
        """Road 12"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("R-5:e-out", "I-27:w-in", (s,c), nb.get_weight(699, 50), LaneType.CITY_ROAD),
                StraightPath("I-27:w-out", "R-5:e-in", (n,c), nb.get_weight(699, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 27"""
        nb.add_intersection(
            "I-27",
            {
                nb.CardinalDirection.NORTH : [1816, 2780],
                nb.CardinalDirection.SOUTH : [1820, 2792],
                nb.CardinalDirection.WEST  : [1812, 2788],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 50)
        )
        

        """Road 9"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-26:n-out", "I-27:s-in", (s,c), nb.get_weight(728, 50), LaneType.CITY_ROAD),
                StraightPath("I-27:s-out", "T-26:n-in", (n,c), nb.get_weight(618, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Roundabout 3"""
        nb.add_roundabout(
            "R-3",
            {
                nb.CardinalDirection.NORTH : [1816, 1963],
                nb.CardinalDirection.SOUTH : [1820, 2005],
                nb.CardinalDirection.WEST  : [1797, 1986],
            },
            nb.get_weight(90, 50)
        )
        

        """Road 24"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-27:n-out", "R-3:s-in", (s,c), nb.get_weight(775, 50), LaneType.CITY_ROAD),
                StraightPath("R-3:s-out", "I-27:n-in", (n,c), nb.get_weight(775, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Turn 17"""
        nb.add_multiple_nodes({
            "T-17:s-out" : [778, 2358],
            "T-17:s-in"  : [782, 2358],
            "T-17:e-in"  : [802, 2334],
            "T-17:e-out" : [802, 2338],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-17:s-in", "T-17:e-out",   0, medium_radius,              right_turn, (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-17:e-in", "T-17:s-out", -90, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Road 20"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-26:n-out", "T-17:s-in", (s,c), nb.get_weight(422, 50), LaneType.CITY_ROAD),
                StraightPath("T-17:s-out", "I-26:n-in", (n,c), nb.get_weight(422, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Intersection 23"""
        nb.add_intersection(
            "I-23",
            {
                nb.CardinalDirection.NORTH : [928, 2330],
                nb.CardinalDirection.WEST  : [924, 2338],
                nb.CardinalDirection.EAST  : [936, 2334],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        

        """Road 21"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-17:e-out", "I-23:w-in", (s,c), nb.get_weight(122, 50), LaneType.CITY_ROAD),
                StraightPath("I-23:w-out", "T-17:e-in", (n,c), nb.get_weight(122, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 19"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("R-5:n-out", "T-18:s-in", (s,c), nb.get_weight(447, 50), LaneType.CITY_ROAD),
                StraightPath("T-18:s-out", "R-5:n-in", (n,c), nb.get_weight(447, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Turn 18"""
        nb.add_multiple_nodes({
            "T-18:s-out" : [1090, 2358],
            "T-18:s-in"  : [1094, 2358],
            "T-18:w-in"  : [1070, 2338],
            "T-18:w-out" : [1070, 2334],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-18:s-in", "T-18:w-out",  0, medium_radius + lane_width, left_turn,  (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-18:w-in", "T-18:s-out", 90, medium_radius,              right_turn, (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Road 22"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-18:w-out", "I-23:e-in", (s,c), nb.get_weight(134, 50), LaneType.CITY_ROAD),
                StraightPath("I-23:e-out", "T-18:w-in", (n,c), nb.get_weight(134, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 23"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-23:n-out", "T-16:s-in", (s,c), nb.get_weight(146, 50), LaneType.CITY_ROAD),
                StraightPath("T-16:s-out", "I-23:n-in", (n,c), nb.get_weight(146, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Turn 16"""
        nb.add_multiple_nodes({
            "T-16:s-out" : [928, 2184],
            "T-16:s-in"  : [932, 2184],
            "T-16:e-in"  : [952, 2160],
            "T-16:e-out" : [952, 2164],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-16:s-in", "T-16:e-out",   0, medium_radius,              right_turn, (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-16:e-in", "T-16:s-out", -90, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Road 27"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-16:e-out", "T-27:w-in", (s,c), nb.get_weight(706, 50), LaneType.CITY_ROAD),
                StraightPath("T-27:w-out", "T-16:e-in", (n,c), nb.get_weight(706, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Turn 27"""
        nb.add_multiple_nodes({
            "T-27:w-out" : [1658, 2160],
            "T-27:w-in"  : [1658, 2164],
            "T-27:n-in"  : [1678, 2140],
            "T-27:n-out" : [1682, 2140],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-27:w-in", "T-27:n-out",  90, medium_radius + lane_width, left_turn,  (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-27:n-in", "T-27:w-out", 180, medium_radius,              right_turn, (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 28"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-27:n-out", "I-22:s-in", (s,c), nb.get_weight(150, 50), LaneType.CITY_ROAD),
                StraightPath("I-22:s-out", "T-27:n-in", (n,c), nb.get_weight(150, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Intersection 22"""
        nb.add_intersection(
            "I-22",
            {
                nb.CardinalDirection.SOUTH : [1682, 1990],
                nb.CardinalDirection.WEST  : [1674, 1986],
                nb.CardinalDirection.EAST  : [1686, 1982],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 29"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-22:e-out", "R-3:w-in", (s,c), nb.get_weight(111, 50), LaneType.CITY_ROAD),
                StraightPath("R-3:w-out", "I-22:e-in", (n,c), nb.get_weight(111, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 18"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-24:n-out", "I-2:s-in", (s,c), nb.get_weight(550, 80), LaneType.ROAD),
                StraightPath("I-2:s-out", "I-24:n-in", (n,c), nb.get_weight(550, 80), LaneType.ROAD),
            ]
        })
        
        
        """Intersection 2"""
        nb.add_intersection(
            "I-2",
            {
                nb.CardinalDirection.NORTH : [4, 2190],
                nb.CardinalDirection.SOUTH : [8, 2202],
                nb.CardinalDirection.WEST  : [0, 2198],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 80)
        )
        
        
        """Road 79"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-2:w-out", "R-4:e-in", (s,c), nb.get_weight(185, 80), LaneType.ROAD),
                StraightPath("R-4:e-out", "I-2:w-in", (n,c), nb.get_weight(185, 80), LaneType.ROAD),
            ]
        })
        
        
        """Road 26"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-2:n-out", "I-20:s-in", (s,c), nb.get_weight(200, 80), LaneType.ROAD),
                StraightPath("I-20:s-out", "I-2:n-in", (n,c), nb.get_weight(200, 80), LaneType.ROAD),
            ]
        })
        
        
        """Intersection 20"""
        nb.add_intersection(
            "I-20",
            {
                nb.CardinalDirection.NORTH : [4, 1978],
                nb.CardinalDirection.SOUTH : [8, 1990],
                nb.CardinalDirection.EAST  : [12, 1982],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 80)
        )
        
        
        """Road 31"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-20:e-out", "I-21:w-in", (s,c), nb.get_weight(200, 50), LaneType.CITY_ROAD),
                StraightPath("I-21:w-out", "I-20:e-in", (n,c), nb.get_weight(200, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Intersection 21"""
        nb.add_intersection(
            "I-21",
            {
                nb.CardinalDirection.NORTH : [216, 1978],
                nb.CardinalDirection.WEST  : [212, 1986],
                nb.CardinalDirection.EAST  : [224, 1982],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 30"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-21:e-out", "I-22:w-in", (s,c), nb.get_weight(1450, 50), LaneType.CITY_ROAD),
                StraightPath("I-22:w-out", "I-21:e-in", (n,c), nb.get_weight(1450, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 25"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("R-4:n-out", "T-12:s-in", (s,c), nb.get_weight(793, 80), LaneType.ROAD),
                StraightPath("T-12:s-out", "R-4:n-in", (n,c), nb.get_weight(793, 80), LaneType.ROAD),
            ]
        })
        
        
        """Turn 12"""
        nb.add_multiple_nodes({
            "T-12:s-out" : [-208, 1382],
            "T-12:s-in"  : [-204, 1382],
            "T-12:e-in"  : [-184, 1358],
            "T-12:e-out" : [-184, 1362],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-12:s-in", "T-12:e-out",   0, medium_radius,              right_turn, (s,c), nb.get_weight(31, 80), LaneType.ROAD),
                CircularPath("T-12:e-in", "T-12:s-out", -90, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 80), LaneType.ROAD),
            ]
        })
        

        """Road 39"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-12:e-out", "R-1:w-in", (s,c), nb.get_weight(172, 80), LaneType.ROAD),
                StraightPath("R-1:w-out", "T-12:e-in", (n,c), nb.get_weight(72, 80), LaneType.ROAD),
            ]
        })
        
        
        """Roundabout 1"""
        nb.add_roundabout(
            "R-1",
            {
                nb.CardinalDirection.NORTH : [4, 1339],
                nb.CardinalDirection.SOUTH : [8, 1381],
                nb.CardinalDirection.WEST  : [-15, 1362],
                nb.CardinalDirection.EAST  : [27, 1358],
            },
            nb.get_weight(90, 80)
        )
        
        
        """Road 34"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-20:n-out", "R-1:s-in", (s,c), nb.get_weight(573, 80), LaneType.ROAD),
                StraightPath("R-1:s-out", "I-20:n-in", (n,c), nb.get_weight(573, 80), LaneType.ROAD),
            ]
        })
        
        
        """Road 33"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-21:n-out", "I-19:s-in", (s,c), nb.get_weight(100, 50), LaneType.CITY_ROAD),
                StraightPath("I-19:s-out", "I-21:n-in", (n,c), nb.get_weight(100, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Intersection 19"""
        nb.add_intersection(
            "I-19",
            {
                nb.CardinalDirection.NORTH : [216, 1866],
                nb.CardinalDirection.SOUTH : [220, 1878],
                nb.CardinalDirection.WEST  : [212, 1874],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 50)
        )
        
        
        """Road 76"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-19:w-out", "T-15:e-in", (s,c), nb.get_weight(122, 50), LaneType.CITY_ROAD),
                StraightPath("T-15:e-out", "I-19:w-in", (n,c), nb.get_weight(122, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Turn 15"""
        nb.add_multiple_nodes({
            "T-15:n-out" : [70, 1850],
            "T-15:n-in"  : [66, 1850],
            "T-15:e-in"  : [90, 1870],
            "T-15:e-out" : [90, 1874],
        })
        
        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-15:e-in", "T-15:n-out", -90, medium_radius,              right_turn, (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-15:n-in", "T-15:e-out", 180, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Intersection 14"""
        nb.add_intersection(
            "I-14",
            {
                nb.CardinalDirection.SOUTH : [70, 1366],
                nb.CardinalDirection.WEST  : [62, 1362],
                nb.CardinalDirection.EAST  : [74, 1358],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 36"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-15:n-out", "I-14:s-in", (s,c), nb.get_weight(480, 50), LaneType.CITY_ROAD),
                StraightPath("I-14:s-out", "T-15:n-in", (n,c), nb.get_weight(480, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 45"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("R-1:e-out", "I-14:w-in", (s,c), nb.get_weight(35, 50), LaneType.CITY_ROAD),
                StraightPath("I-14:w-out", "R-1:e-in", (n,c), nb.get_weight(35, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 35"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-19:n-out", "T-14:s-in", (s,c), nb.get_weight(364, 50), LaneType.CITY_ROAD),
                StraightPath("T-14:s-out", "I-19:n-in", (n,c), nb.get_weight(364, 50), LaneType.CITY_ROAD),
            ]
        })


        """Turn 14"""
        nb.add_multiple_nodes({
            "T-14:s-out" : [216, 1502],
            "T-14:s-in"  : [220, 1502],
            "T-14:e-in"  : [240, 1478],
            "T-14:e-out" : [240, 1482],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-14:s-in", "T-14:e-out",   0, medium_radius,              right_turn, (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-14:e-in", "T-14:s-out", -90, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })


        """Road 38"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-14:e-out", "I-18:w-in", (s,c), nb.get_weight(1414, 50), LaneType.CITY_ROAD),
                StraightPath("I-18:w-out", "T-14:e-in", (n,c), nb.get_weight(1414, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 18"""
        nb.add_intersection(
            "I-18",
            {
                nb.CardinalDirection.NORTH : [1658, 1474],
                nb.CardinalDirection.WEST  : [1654, 1482],
                nb.CardinalDirection.EAST  : [1666, 1478],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        

        """Road 37"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-18:e-out", "R-2:w-in", (s,c), nb.get_weight(131, 50), LaneType.CITY_ROAD),
                StraightPath("R-2:w-out", "I-18:e-in", (n,c), nb.get_weight(131, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Roundabout 2"""
        nb.add_roundabout(
            "R-2",
            {
                nb.CardinalDirection.NORTH : [1816, 1459],
                nb.CardinalDirection.SOUTH : [1820, 1501],
                nb.CardinalDirection.WEST  : [1797, 1482],
            },
            nb.get_weight(90, 80)
        )
        

        """Road 37"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("R-3:n-out", "R-2:s-in", (s,c), nb.get_weight(462, 50), LaneType.CITY_ROAD),
                StraightPath("R-2:s-out", "R-3:n-in", (n,c), nb.get_weight(462, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 40"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-18:n-out", "T-13:s-in", (s,c), nb.get_weight(92, 50), LaneType.CITY_ROAD),
                StraightPath("T-13:s-out", "I-18:n-in", (n,c), nb.get_weight(92, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Turn 13"""
        nb.add_multiple_nodes({
            "T-13:s-out" : [1658, 1382],
            "T-13:s-in"  : [1662, 1382],
            "T-13:w-in"  : [1638, 1362],
            "T-13:w-out" : [1638, 1358],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-13:s-in", "T-13:w-out",  0, medium_radius + lane_width, left_turn,  (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-13:w-in", "T-13:s-out", 90, medium_radius,              right_turn, (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })

        
        """Road 41"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-13:w-out", "I-17:e-in", (s,c), nb.get_weight(118, 50), LaneType.CITY_ROAD),
                StraightPath("I-17:e-out", "T-13:w-in", (n,c), nb.get_weight(118, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 17"""
        nb.add_intersection(
            "I-17",
            {
                nb.CardinalDirection.NORTH : [1512, 1354],
                nb.CardinalDirection.WEST  : [1508, 1362],
                nb.CardinalDirection.EAST  : [1520, 1358],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 42"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-17:w-out", "I-16:e-in", (s,c), nb.get_weight(200, 50), LaneType.CITY_ROAD),
                StraightPath("I-16:e-out", "I-17:w-in", (n,c), nb.get_weight(200, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 16"""
        nb.add_intersection(
            "I-16",
            {
                nb.CardinalDirection.NORTH : [1236, 1354],
                nb.CardinalDirection.WEST  : [1232, 1362],
                nb.CardinalDirection.EAST  : [1244, 1358],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 43"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-16:w-out", "I-15:e-in", (s,c), nb.get_weight(750, 50), LaneType.CITY_ROAD),
                StraightPath("I-15:e-out", "I-16:w-in", (n,c), nb.get_weight(750, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 15"""
        nb.add_intersection(
            "I-15",
            {
                nb.CardinalDirection.NORTH : [474, 1354],
                nb.CardinalDirection.WEST  : [470, 1362],
                nb.CardinalDirection.EAST  : [482, 1358],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 44"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-15:w-out", "I-14:e-in", (s,c), nb.get_weight(400, 50), LaneType.CITY_ROAD),
                StraightPath("I-14:e-out", "I-15:w-in", (n,c), nb.get_weight(400, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 46"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-17:n-out", "I-13:s-in", (s,c), nb.get_weight(152, 50), LaneType.CITY_ROAD),
                StraightPath("I-13:s-out", "I-17:n-in", (n,c), nb.get_weight(152, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 13"""
        nb.add_intersection(
            "I-13",
            {
                nb.CardinalDirection.NORTH : [1512, 1190],
                nb.CardinalDirection.SOUTH : [1516, 1202],
                nb.CardinalDirection.EAST  : [1520, 1194],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 50)
        )
        
        
        """Road 77"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-13:e-out", "T-11:w-in", (s,c), nb.get_weight(74, 50), LaneType.CITY_ROAD),
                StraightPath("T-11:w-out", "I-13:e-in", (n,c), nb.get_weight(74, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Turn 11"""
        nb.add_multiple_nodes({
            "T-11:w-out" : [1554, 1194],
            "T-11:w-in"  : [1554, 1198],
            "T-11:n-in"  : [1574, 1174],
            "T-11:n-out" : [1578, 1174],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-11:w-in", "T-11:n-out",  90, medium_radius + lane_width, left_turn,  (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-11:n-in", "T-11:w-out", 180, medium_radius,              right_turn, (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })


        """Road 52"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-11:n-out", "I-12:s-in", (s,c), nb.get_weight(38, 50), LaneType.CITY_ROAD),
                StraightPath("I-12:s-out", "T-11:n-in", (n,c), nb.get_weight(38, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 12"""
        nb.add_intersection(
            "I-12",
            {
                nb.CardinalDirection.SOUTH : [1578, 1040],
                nb.CardinalDirection.WEST  : [1570, 1036],
                nb.CardinalDirection.EAST  : [1582, 1032],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 53"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-13:n-out", "I-11:s-in", (s,c), nb.get_weight(150, 50), LaneType.CITY_ROAD),
                StraightPath("I-11:s-out", "I-13:n-in", (n,c), nb.get_weight(150, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 11"""
        nb.add_intersection(
            "I-11",
            {
                nb.CardinalDirection.SOUTH : [1516, 1040],
                nb.CardinalDirection.WEST  : [1508, 1036],
                nb.CardinalDirection.EAST  : [1520, 1032],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 55"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-11:e-out", "I-12:w-in", (s,c), nb.get_weight(50, 50), LaneType.CITY_ROAD),
                StraightPath("I-12:w-out", "I-11:e-in", (n,c), nb.get_weight(50, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 47"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-16:n-out", "T-9:s-in", (s,c), nb.get_weight(298, 50), LaneType.CITY_ROAD),
                StraightPath("T-9:s-out", "I-16:n-in", (n,c), nb.get_weight(298, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Turn 9"""
        nb.add_multiple_nodes({
            "T-9:s-out" : [1236, 1056],
            "T-9:s-in"  : [1240, 1056],
            "T-9:e-in"  : [1260, 1032],
            "T-9:e-out" : [1260, 1036],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-9:s-in", "T-9:e-out",   0, medium_radius,              right_turn, (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-9:e-in", "T-9:s-out", -90, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })

        
        """Road 80"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-9:e-out", "I-10:w-in", (s,c), nb.get_weight(86, 50), LaneType.CITY_ROAD),
                StraightPath("I-10:w-out", "T-9:e-in", (n,c), nb.get_weight(86, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 10"""
        nb.add_intersection(
            "I-10",
            {
                nb.CardinalDirection.NORTH : [1350, 1028],
                nb.CardinalDirection.WEST  : [1346, 1036],
                nb.CardinalDirection.EAST  : [1358, 1032],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 56"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-10:e-out", "I-11:w-in", (s,c), nb.get_weight(150, 50), LaneType.CITY_ROAD),
                StraightPath("I-11:w-out", "I-10:e-in", (n,c), nb.get_weight(150, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 54"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-12:e-out", "T-10:w-in", (s,c), nb.get_weight(204, 50), LaneType.CITY_ROAD),
                StraightPath("T-10:w-out", "I-12:e-in", (n,c), nb.get_weight(204, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Turn 10"""
        nb.add_multiple_nodes({
            "T-10:w-out" : [1746, 1032],
            "T-10:w-in"  : [1746, 1036],
            "T-10:n-in"  : [1766, 1012],
            "T-10:n-out" : [1770, 1012],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-10:w-in", "T-10:n-out",  90, medium_radius + lane_width, left_turn,  (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-10:n-in", "T-10:w-out", 180, medium_radius,              right_turn, (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })


        """Road 78"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-10:n-out", "I-8:s-in", (s,c), nb.get_weight(34, 50), LaneType.CITY_ROAD),
                StraightPath("I-8:s-out", "T-10:n-in", (n,c), nb.get_weight(34, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 8"""
        nb.add_intersection(
            "I-8",
            {
                nb.CardinalDirection.SOUTH : [1770, 978],
                nb.CardinalDirection.WEST  : [1762, 974],
                nb.CardinalDirection.EAST  : [1774, 970],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 59"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-8:e-out", "I-9:w-in", (s,c), nb.get_weight(38, 50), LaneType.CITY_ROAD),
                StraightPath("I-9:w-out", "I-8:e-in", (n,c), nb.get_weight(38, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 9"""
        nb.add_intersection(
            "I-9",
            {
                nb.CardinalDirection.NORTH : [1816, 966],
                nb.CardinalDirection.SOUTH : [1820, 978],
                nb.CardinalDirection.WEST  : [1812, 974],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 50)
        )
        

        """Road 51"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("R-2:n-out", "I-9:s-in", (s,c), nb.get_weight(481, 50), LaneType.CITY_ROAD),
                StraightPath("I-9:s-out", "R-2:n-in", (n,c), nb.get_weight(481, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 60"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-8:w-out", "T-6:e-in", (s,c), nb.get_weight(226, 50), LaneType.CITY_ROAD),
                StraightPath("T-6:e-out", "I-8:w-in", (n,c), nb.get_weight(226, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Turn 6"""
        nb.add_multiple_nodes({
            "T-6:e-out" : [1536, 974],
            "T-6:e-in"  : [1536, 970],
            "T-6:n-in"  : [1512, 950],
            "T-6:n-out" : [1516, 950],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-6:e-in", "T-6:n-out",  -90, medium_radius,             right_turn, (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-6:n-in", "T-6:e-out", 180, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })

        
        """Road 62"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-6:n-out", "I-6:s-in", (s,c), nb.get_weight(226, 50), LaneType.CITY_ROAD),
                StraightPath("I-6:s-out", "T-6:n-in", (n,c), nb.get_weight(226, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 6"""
        nb.add_intersection(
            "I-6",
            {
                nb.CardinalDirection.NORTH : [1512, 712],
                nb.CardinalDirection.SOUTH : [1516, 724],
                nb.CardinalDirection.WEST  : [1508, 720],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 50)
        )
        
        
        """Road 63"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-6:w-out", "I-7:e-in", (s,c), nb.get_weight(150, 50), LaneType.CITY_ROAD),
                StraightPath("I-7:e-out", "I-6:w-in", (n,c), nb.get_weight(150, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 7"""
        nb.add_intersection(
            "I-7",
            {
                nb.CardinalDirection.SOUTH : [1354, 724],
                nb.CardinalDirection.WEST  : [1346, 720],
                nb.CardinalDirection.EAST  : [1358, 716],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 57"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-10:n-out", "I-7:s-in", (s,c), nb.get_weight(304, 50), LaneType.CITY_ROAD),
                StraightPath("I-7:s-out", "I-10:n-in", (n,c), nb.get_weight(304, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 48"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-15:n-out", "T-7:s-in", (s,c), nb.get_weight(226, 50), LaneType.CITY_ROAD),
                StraightPath("T-7:s-out", "I-15:n-in", (n,c), nb.get_weight(226, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Turn 7"""
        nb.add_multiple_nodes({
            "T-7:s-out" : [474, 1069],
            "T-7:s-in"  : [478, 1069],
            "T-7:e-in"  : [498, 1045],
            "T-7:e-out" : [498, 1049],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-7:s-in", "T-7:e-out",   0, medium_radius,              right_turn, (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-7:e-in", "T-7:s-out", -90, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 50"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-7:e-out", "T-8:w-in", (s,c), nb.get_weight(406, 50), LaneType.CITY_ROAD),
                StraightPath("T-8:w-out", "T-7:e-in", (n,c), nb.get_weight(406, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Turn 8"""
        nb.add_multiple_nodes({
            "T-8:w-out" : [904, 1045],
            "T-8:w-in"  : [904, 1049],
            "T-8:n-in"  : [924, 1025],
            "T-8:n-out" : [928, 1025],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-8:w-in", "T-8:n-out",  90, medium_radius + lane_width, left_turn,  (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-8:n-in", "T-8:w-out", 180, medium_radius,              right_turn, (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })

        
        """Road 58"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-8:n-out", "T-5:s-in", (s,c), nb.get_weight(285, 50), LaneType.CITY_ROAD),
                StraightPath("T-5:s-out", "T-8:n-in", (n,c), nb.get_weight(285, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Turn 5"""
        nb.add_multiple_nodes({
            "T-5:s-out" : [924, 740],
            "T-5:s-in"  : [928, 740],
            "T-5:e-in"  : [948, 716],
            "T-5:e-out" : [948, 720],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-5:s-in", "T-5:e-out",   0, medium_radius,              right_turn, (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-5:e-in", "T-5:s-out", -90, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 64"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-5:e-out", "I-7:w-in", (s,c), nb.get_weight(285, 50), LaneType.CITY_ROAD),
                StraightPath("I-7:w-out", "T-5:e-in", (n,c), nb.get_weight(285, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 65"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-6:n-out", "T-4:s-in", (s,c), nb.get_weight(178, 50), LaneType.CITY_ROAD),
                StraightPath("T-4:s-out", "I-6:n-in", (n,c), nb.get_weight(178, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Turn 4"""
        nb.add_multiple_nodes({
            "T-4:s-out" : [1512, 534],
            "T-4:s-in"  : [1516, 534],
            "T-4:w-in"  : [1492, 514],
            "T-4:w-out" : [1492, 510],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-4:s-in", "T-4:w-out",  0, medium_radius + lane_width, left_turn,  (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-4:w-in", "T-4:s-out", 90, medium_radius,              right_turn, (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 66"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-4:w-out", "T-3:e-in", (s,c), nb.get_weight(1198, 50), LaneType.CITY_ROAD),
                StraightPath("T-3:e-out", "T-4:w-in", (n,c), nb.get_weight(1198, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Turn 3"""
        nb.add_multiple_nodes({
            "T-3:e-out" : [294, 514],
            "T-3:e-in"  : [294, 510],
            "T-3:n-in"  : [270, 490],
            "T-3:n-out" : [274, 490],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-3:e-in", "T-3:n-out", -90, medium_radius,              right_turn, (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-3:n-in", "T-3:e-out", 180, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 68"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-3:n-out", "I-4:s-in", (s,c), nb.get_weight(174, 50), LaneType.CITY_ROAD),
                StraightPath("I-4:s-out", "T-3:n-in", (n,c), nb.get_weight(174, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Intersection 4"""
        nb.add_intersection(
            "I-4",
            {
                nb.CardinalDirection.SOUTH : [273, 316],
                nb.CardinalDirection.WEST  : [266, 312],
                nb.CardinalDirection.EAST  : [278, 308],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 50)
        )
        
        
        """Road 71"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-4:e-out", "T-2:w-in", (s,c), nb.get_weight(292, 50), LaneType.CITY_ROAD),
                StraightPath("T-2:w-out", "I-4:e-in", (n,c), nb.get_weight(292, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Turn 2"""
        nb.add_multiple_nodes({
            "T-2:w-out" : [600, 308],
            "T-2:w-in"  : [600, 312],
            "T-2:s-in"  : [624, 332],
            "T-2:s-out" : [620, 332],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-2:w-in", "T-2:s-out", 90, medium_radius,              right_turn, (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-2:s-in", "T-2:w-out",  0, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 69"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-2:s-out", "T-28:n-in", (s,c), nb.get_weight(106, 50), LaneType.CITY_ROAD),
                StraightPath("T-28:n-out", "T-2:s-in", (n,c), nb.get_weight(106, 50), LaneType.CITY_ROAD),
            ]
        })
        

        """Turn 27"""
        nb.add_multiple_nodes({
            "T-28:n-out" : [624, 438],
            "T-28:n-in"  : [620, 438],
            "T-28:e-in"  : [644, 458],
            "T-28:e-out" : [644, 462],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-28:n-in", "T-28:e-out", 180, medium_radius + lane_width, left_turn,   (s,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
                CircularPath("T-28:e-in", "T-28:n-out",  -90, medium_radius,             right_turn,  (n,c), nb.get_weight(31, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 67"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-28:e-out", "I-5:w-in", (s,c), nb.get_weight(1168, 50), LaneType.CITY_ROAD),
                StraightPath("I-5:w-out", "T-28:e-in", (n,c), nb.get_weight(1168, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Intersection 5"""
        nb.add_intersection(
            "I-5",
            {
                nb.CardinalDirection.NORTH : [1816, 454],
                nb.CardinalDirection.SOUTH : [1820, 466],
                nb.CardinalDirection.WEST  : [1812, 462],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 50)
        )
        
        
        """Road 61"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-9:n-out", "I-5:s-in", (s,c), nb.get_weight(500, 50), LaneType.CITY_ROAD),
                StraightPath("I-5:s-out", "I-9:n-in", (n,c), nb.get_weight(500, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 49"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("R-1:n-out", "I-3:s-in", (s,c), nb.get_weight(1023, 80), LaneType.ROAD),
                StraightPath("I-3:s-out", "R-1:n-in", (n,c), nb.get_weight(1023, 80), LaneType.ROAD),
            ]
        })
        
        
        """Intersection 3"""
        nb.add_intersection(
            "I-3",
            {
                nb.CardinalDirection.NORTH : [4, 304],
                nb.CardinalDirection.SOUTH : [8, 316],
                nb.CardinalDirection.EAST  : [12, 308],
            },
            nb.PathPriority.NORTH_SOUTH,
            nb.get_weight(12, 80)
        )
        
        
        """Road 72"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-3:e-out", "I-4:w-in", (s,c), nb.get_weight(254, 50), LaneType.CITY_ROAD),
                StraightPath("I-4:w-out", "I-3:e-in", (n,c), nb.get_weight(254, 50), LaneType.CITY_ROAD),
            ]
        })
        
        
        """Road 73"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-3:n-out", "I-1:s-in", (s,c), nb.get_weight(300, 80), LaneType.ROAD),
                StraightPath("I-1:s-out", "I-3:n-in", (n,c), nb.get_weight(300, 80), LaneType.ROAD),
            ]
        })
        
        
        """Intersection 1"""
        nb.add_intersection(
            "I-1",
            {
                nb.CardinalDirection.SOUTH : [8, 4],
                nb.CardinalDirection.WEST  : [0, 0],
                nb.CardinalDirection.EAST  : [12, -4],
            },
            nb.PathPriority.EAST_WEST,
            nb.get_weight(12, 80)
        )
        
        
        """Road 74"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-1:e-out", "T-1:w-in", (s,c), nb.get_weight(1784, 80), LaneType.ROAD),
                StraightPath("T-1:w-out", "I-1:e-in", (n,c), nb.get_weight(1784, 80), LaneType.ROAD),
            ]
        })
        

        """Turn 1"""
        nb.add_multiple_nodes({
            "T-1:w-out" : [1796, -4],
            "T-1:w-in"  : [1796, 0],
            "T-1:s-in"  : [1820, 20],
            "T-1:s-out" : [1816, 20],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-1:w-in", "T-1:s-out", 90, medium_radius,              right_turn, (s,c), nb.get_weight(31, 80), LaneType.ROAD),
                CircularPath("T-1:s-in", "T-1:w-out",  0, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 80), LaneType.ROAD),
            ]
        })
        
        
        """Road 70"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-1:s-out", "I-5:n-in", (s,c), nb.get_weight(434, 80), LaneType.ROAD),
                StraightPath("I-5:n-out", "T-1:s-in", (n,c), nb.get_weight(434, 80), LaneType.ROAD),
            ]
        })
        
        
        """Road 75"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("I-1:w-out", "T-29:e-in", (s,c), nb.get_weight(300, 80), LaneType.ROAD),
                StraightPath("T-29:e-out", "I-1:w-in", (n,c), nb.get_weight(300, 80), LaneType.ROAD),
            ]
        })
        

        """Turn 29"""
        nb.add_multiple_nodes({
            "T-29:e-out" : [-300, 0],
            "T-29:e-in"  : [-300, -4],
            "T-29:s-in"  : [-320, 20],
            "T-29:s-out" : [-324, 20],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-29:e-in", "T-29:s-out", -90, medium_radius + lane_width, left_turn,  (s,c), nb.get_weight(31, 80), LaneType.ROAD),
                CircularPath("T-29:s-in", "T-29:e-out",   0, medium_radius,              right_turn, (n,c), nb.get_weight(31, 80), LaneType.ROAD),
            ]
        })
        
        
        """Turn 30"""
        nb.add_multiple_nodes({
            "T-30:e-out" : [-300, 3410],
            "T-30:e-in"  : [-300, 3406],
            "T-30:n-in"  : [-324, 3386],
            "T-30:n-out" : [-320, 3386],
        })

        nb.add_multiple_paths({
            nb.PathType.CIRCULAR : [
                CircularPath("T-30:e-in", "T-30:n-out", -90, medium_radius,              right_turn, (s,c), nb.get_weight(31, 80), LaneType.ROAD),
                CircularPath("T-30:n-in", "T-30:e-out", 180, medium_radius + lane_width, left_turn,  (n,c), nb.get_weight(31, 80), LaneType.ROAD),
            ]
        })
        
        
        """Highway 1"""
        nb.add_multiple_nodes({
            "H-1:n-in:1"  : [-328, 30],
            "H-1:n-in:2"  : [-324, 30],
            "H-1:n-out:2" : [-320, 30],
            "H-1:n-out:1" : [-316, 30],
            
            "H-1:s-out:1" : [-328, 3376],
            "H-1:s-out:2" : [-324, 3376],
            "H-1:s-in:2"  : [-320, 3376],
            "H-1:s-in:1"  : [-316, 3376],
        })
        
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("T-30:n-out", "H-1:s-in:1", (n,c), nb.get_weight(10, 130), LaneType.HIGHWAY),
                StraightPath("T-30:n-out", "H-1:s-in:2", (c,s), nb.get_weight(10, 130), LaneType.HIGHWAY),
                StraightPath("H-1:s-out:1", "T-30:n-in", (n,c), nb.get_weight(10, 130), LaneType.HIGHWAY),
                StraightPath("H-1:s-out:2", "T-30:n-in", (c,s), nb.get_weight(10, 130), LaneType.HIGHWAY),
                
                StraightPath("T-29:s-out", "H-1:n-in:1", (n,c), nb.get_weight(10, 130), LaneType.HIGHWAY),
                StraightPath("T-29:s-out", "H-1:n-in:2", (c,s), nb.get_weight(10, 130), LaneType.HIGHWAY),
                StraightPath("H-1:n-out:1", "T-29:s-in", (n,c), nb.get_weight(10, 130), LaneType.HIGHWAY),
                StraightPath("H-1:n-out:2", "T-29:s-in", (c,s), nb.get_weight(10, 130), LaneType.HIGHWAY),

                StraightPath("H-1:s-in:1", "H-1:n-out:1", (n,c), nb.get_weight(14, 130), LaneType.HIGHWAY),
                StraightPath("H-1:s-in:2", "H-1:n-out:2", (c,s), nb.get_weight(14, 130), LaneType.HIGHWAY),
                StraightPath("H-1:n-in:2", "H-1:s-out:2", (c,s), nb.get_weight(14, 130), LaneType.HIGHWAY),
                StraightPath("H-1:n-in:1", "H-1:s-out:1", (n,c), nb.get_weight(14, 130), LaneType.HIGHWAY),
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

    def _get_shortest_path(self, startpoint: tuple[str, str, int], destination: tuple[str, str, int]) -> list[tuple[str, str, int]]:
        shortest_path = [startpoint[0]] + self.shortest_paths[startpoint[1]][destination[1]]
        return [(shortest_path[i], shortest_path[i+1], None) for i in range(len(shortest_path)-1)]

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
        exit_edge  = self._get_random_destination_different_from(entry_edge) 

        while (entry_edge[0] in exit_edge or entry_edge[1] in exit_edge):
            logger.info(f"\t_spawn_vehicle                         :: Element in 'entry_edge' was in 'exit_edge' -- {entry_edge} ~> {exit_edge}")
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

        routes_logger.info(f"\t_spawn_vehicele :: planning route {entry_edge} ~> {exit_edge}")
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
            close_edge = self.road.network.get_closest_lane_index(edge_lane.position(0,0), edge_lane.heading_at(60))
            if edge != close_edge:
                logger.info(f"\t_categorize_edges_by_type          :: close_edge != edge -- {close_edge} : {edge}")
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
        categories: list[list[tuple[str, str, int]]] = [self.I_edges, self.R_edges, self.H_edges, self.T_edges]
        category_index: int                          = self.episode_count % len(categories)
        chosen_category: list[tuple[str, str, int]]  = categories[category_index]

        edge = self.get_random_edge_from(chosen_category)
        
        # Validate the edge can be found again
        edge_lane = self.road.network.get_lane(edge)
        close_edge = self.road.network.get_closest_lane_index(edge_lane.position(0,0), edge_lane.heading_at(60))

        while edge != close_edge:
            logger.info(f"\t_get_balanced_random_edge              :: close_edge != edge -- {close_edge} : {edge}")
            edge = self.get_random_edge_from(chosen_category)
            edge_lane = self.road.network.get_lane(edge)
            close_edge = self.road.network.get_closest_lane_index(edge_lane.position(0,0), edge_lane.heading_at(60))

        return edge
        
    def _get_random_edge(self) -> tuple[str, str, int]:
        edges = list(self.local_graph_net.edges)
        edge = self.get_random_edge_from(edges)
        
        # Validate the edge can be found again
        edge_lane = self.road.network.get_lane(edge)
        close_edge = self.road.network.get_closest_lane_index(edge_lane.position(0,0), edge_lane.heading_at(60))

        while edge != close_edge:
            logger.info(f"\t_get_random_edge                       :: close_edge != edge -- {close_edge} : {edge}")
            self.local_graph_net.remove_edge(*edge)

            edge = self.get_random_edge_from(edges)
            edge_lane = self.road.network.get_lane(edge)
            close_edge = self.road.network.get_closest_lane_index(edge_lane.position(0,0), edge_lane.heading_at(60))
        
        return edge
    
    def _get_random_destination_different_from(self, start_edge) -> tuple[str, str, int]:
        destination = self._get_random_edge()
        
        # Validate that no vertex from 'start_edge' is in 'destination'
        while (start_edge[0] in destination or start_edge[1] in destination):
            logger.info(f"\t_get_random_destination_different_from :: Element in 'startpoint' was in 'destination' -- {start_edge} ~> {destination}")
            destination = self._get_random_edge()
            
        return destination
    
    def get_destination_close_to(self, start_edge, ego_destination) -> tuple[str, str, int]:
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
        # FIXME: remove duct-tape
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
                routes_logger.info(f"\t_make_vehicele  :: ego vehicle planning route {startpoint} ~> {destination}")
                # print(f"destination[1]: {destination[1]} :: plan_route_to({destination[1]})")

                ego_vehicle.route = self._get_shortest_path(startpoint, destination)
                # ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                # ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                print("Got an attribute error")
                logger.warning(f"In episode '{self.episode_count}': AttributeError while planning ego route")
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
        
        # ---::: Code to spawn other vehicles goes here :::---

        # Use the ego vehicle's route. This should be a list of edges like [(start_node, end_node, lane_index), ...]
        candidate_edges = list(ego_vehicle.route)  # Already a list of edges along the route

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
            behind_longitudinal = ego_longtitudinal - (behind_spacing * (vehicles_behind_placed + 1))
            if behind_longitudinal < 0:
                break
            
            pos_behind = ego_lane.position(behind_longitudinal, 0)
            heading_behind = ego_lane.heading_at(behind_longitudinal)
            
            if self.road.network.get_lane_type(startpoint[0], startpoint[1]) == LaneType.HIGHWAY:
                speed_behind = np.random.uniform(20, 30)
            else:
                speed_behind = np.random.uniform(5, 20)
                
            other_vehicle = vehicle_type(self.road, pos_behind, heading=heading_behind, speed=speed_behind)

            no_collision = True
            for v in self.road.vehicles:
                if np.linalg.norm(v.position - other_vehicle.position) < 10:
                    no_collision = False
                    break
                
            
            if no_collision:
                # Route for behind vehicle: also try close to ego's final node
                start_edge = startpoint
                if random.random() < close_to_ego_dest_probability:
                    vehicle_destination = self.get_destination_close_to(start_edge, ego_final_node)
                    if vehicle_destination is None:
                        vehicle_destination = self._get_random_destination_different_from(start_edge)
                else:
                    vehicle_destination = self._get_random_destination_different_from(start_edge)

                try:
                    other_vehicle.route = self._get_shortest_path(start_edge, vehicle_destination)
                except Exception as e:
                    logger.warning(f"Could not get shortest path for behind vehicle: {e}")
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
                
                if self.road.network.get_lane_type(edge[0], edge[1]) == LaneType.HIGHWAY:
                    speed = np.random.uniform(20, 30)
                else:
                    speed = np.random.uniform(5, 20)
                
                other_vehicle = vehicle_type(self.road, pos, heading=heading, speed=speed)
                other_vehicle.check_collisions = False
                
                # Check collision with existing vehicles
                no_collision = True
                for v in self.road.vehicles:
                    if np.linalg.norm(v.position - other_vehicle.position) < 10:  # 10 meters safe distance
                        no_collision = False
                        break

                if no_collision:
                    start_edge = edge

                    if random.random() < close_to_ego_dest_probability:
                        # Get destination close to ego_destination
                        vehicle_destination = self.get_destination_close_to(start_edge, ego_final_node)
                        if vehicle_destination is None:
                            vehicle_destination = self._get_random_destination_different_from(start_edge)
                    else:
                        # Get random destination
                        vehicle_destination = self._get_random_destination_different_from(start_edge)
                    
                    try:
                        routes_logger.info(f"\t_make_vehicele  :: planning route {start_edge} ~> {vehicle_destination}")
                        other_vehicle.route = self._get_shortest_path(start_edge, vehicle_destination)
                    except Exception as e:
                        logger.warning(f"Could not get shortest path for other vehicle: {e}")
                        other_vehicle.route = []

                    other_vehicle.randomize_behavior()
                    self.road.vehicles.append(other_vehicle)
                    placed = True
                    break

            if not placed:
                logger.info(f"Could not place a new vehicle without collision after {_attempts} attempts.")

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
        # Add logic for checking if we have reached our destination
        return (
            self.time >= self.config["duration"]
            or self.vehicle.remaining_route_nodes == 0
        )
