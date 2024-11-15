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
from highway_env.road.lanes.lane_utils import LineType
from highway_env.road.lanes.abstract_lanes import AbstractLane

from highway_env.road.regulation import RegulatedRoad
from highway_env.network_builder import NetworkBuilder, StraightPath, CircularPath, Path
from highway_env.road.lanes.unweighted_lanes import StraightLane, SineLane, CircularLane


class HomemadeHighwayRefactor(AbstractEnv):
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

        net = RoadNetwork()
        nb = NetworkBuilder()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        left_turn = False
        right_turn = True
        
        """First highway circuit"""
        nb.add_multiple_nodes({
            "a:1" : [0, 0],
            "a:2" : [0, 4],
            
            "b:1" : [300, 0],
            "b:2" : [300, 4],

            "c:1" : [600, 0],
            "c:2" : [600, 4],
            
            "d:1" : [700, 0],
            "d:2" : [700, 4],
            
            "e:1" : [720, -20],
            "e:2" : [724, -20],
            
            "f:1" : [720, -120],
            "f:2" : [724, -120],
            
            "g:1" : [700, -140],
            "g:2" : [700, -144],
            
            "h:1" : [0, -140],
            "h:2" : [0, -144],
            
            "i:1" : [-20, -120],
            "i:2" : [-24, -120],
            
            "j:1" : [-20, -20],
            "j:2" : [-24, -20],
        })
        
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("a:1", "b:1", (c,s)),
                StraightPath("a:2", "b:2", (n,c)),
                
                StraightPath("b:1", "c:1", (c,s)),
                StraightPath("b:2", "c:2", (n,c)),
                
                StraightPath("c:1", "d:1", (c,s)),
                StraightPath("c:2", "d:2", (n,c)),
                
                StraightPath("e:1", "f:1", (c,s)),
                StraightPath("e:2", "f:2", (n,c)),
                
                StraightPath("g:1", "h:1", (c,s)),
                StraightPath("g:2", "h:2", (n,c)),
                
                StraightPath("i:1", "j:1", (c,s)),
                StraightPath("i:2", "j:2", (n,c)),
            ],
            nb.PathType.CIRCULAR : [
                CircularPath("d:1", "e:1", 90, 20, left_turn, (c,s)),
                CircularPath("d:2", "e:2", 90, 24, left_turn, (n,c)),
                
                CircularPath("f:1", "g:1", 0, 20, left_turn, (c,s)),
                CircularPath("f:2", "g:2", 0, 24, left_turn, (n,c)),
                
                CircularPath("h:1", "i:1", -90, 20, left_turn, (c,s)),
                CircularPath("h:2", "i:2", -90, 24, left_turn, (n,c)),
                
                CircularPath("j:1", "a:1", 180, 20, left_turn, (c,s)),
                CircularPath("j:2", "a:2", 180, 24, left_turn, (n,c)),
            ]
        })
        
        """Second highway circuit"""
        nb.add_multiple_nodes({
            "b:3" : [300, 8],
            "b:4" : [300, 12],
            
            "c:3" : [600, 8],
            "c:4" : [600, 12],
            
            "k:1" : [624, 32],
            "k:2" : [620, 32],
            
            "l:1" : [624, 132],
            "l:2" : [620, 132],
            
            "m:1" : [600, 156],
            "m:2" : [600, 152],
            
            "n:1" : [300, 156],
            "n:2" : [300, 152],
            
            "o:1" : [276, 132],
            "o:2" : [280, 132],
            
            "p:1" : [276, 28],
            "p:2" : [280, 28],
        })
        
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("b:3", "c:3", (c,s)),
                StraightPath("b:4", "c:4", (n,c)),
                
                StraightPath("k:1", "l:1", (c,s)),
                StraightPath("k:2", "l:2", (n,c)),
                
                StraightPath("m:1", "n:1", (c,s)),
                StraightPath("m:2", "n:2", (n,c)),
                
                StraightPath("o:1", "p:1", (c,s)),
                StraightPath("o:2", "p:2", (n,c)),
            ],
            nb.PathType.CIRCULAR : [
                CircularPath("c:3", "k:1", 90, 24, right_turn, (c,s)),
                CircularPath("c:4", "k:2", 90, 20, right_turn, (n,c)),
                
                CircularPath("l:1", "m:1", 180, 24, right_turn, (c,s)),
                CircularPath("l:2", "m:2", 180, 20, right_turn, (n,c)),
                
                CircularPath("n:1", "o:1", -90, 24, right_turn, (c,s)),
                CircularPath("n:2", "o:2", -90, 20, right_turn, (n,c)),
                
                CircularPath("p:1", "b:3", 0, 24, right_turn, (c,s)),
                CircularPath("p:2", "b:4", 0, 20, right_turn, (n,c)),
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

        ego_lane = self.road.network.get_lane(("b", "c", 0)) # This is to place the car on a road between two points
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(150, 0),            # Use the first value to place car down the road
            speed=0,                              # Speed of car
            heading=ego_lane.heading_at(90),      # Use this to change the direction of the car. "0" is north
        )
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