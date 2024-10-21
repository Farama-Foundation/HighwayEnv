from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from highway_env.road.lane import CircularLane, LineType, StraightLane


class HomemadeRacetrack(AbstractEnv):
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
            
    def _make_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""

        line_none, line_continuous, line_striped = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        
        lane1 = (line_continuous, line_none)
        lane2 = (line_striped, line_continuous)
        
        
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane(
            [0, 0],
            [100, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimits[1],
        )
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, 5],
                [100, 5],
                line_types=lane2,
                width=5,
                speed_limit=speedlimits[1],
            ),
        )
        
        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(0),
                width=5,
                clockwise=False,
                line_types=lane1,
                speed_limit=speedlimits[2],
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + 5,
                np.deg2rad(90),
                np.deg2rad(0),
                width=5,
                clockwise=False,
                line_types=lane2,
                speed_limit=speedlimits[2],
            ),
        )
        
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [120, -20],
                [120, -120],
                line_types=lane1,
                width=5,
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [125, -20],
                [125, -120],
                line_types=lane2,
                width=5,
                speed_limit=speedlimits[3],
            ),
        )
        
        center2 = [100,-120]
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii1,
                np.deg2rad(0),
                np.deg2rad(-90),
                width=5,
                clockwise=False,
                line_types=lane1,
                speed_limit=speedlimits[2]
            )
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii1 + 5,
                np.deg2rad(0),
                np.deg2rad(-90),
                width=5,
                clockwise=False,
                line_types=lane2,
                speed_limit=speedlimits[2]
            )
        )
        
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [100, -140],
                [0, -140],
                line_types=lane1,
                width=5,
                speed_limit=speedlimits[2]
            )
        )
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [100, -145],
                [0, -145],
                line_types=lane2,
                width=5,
                speed_limit=speedlimits[2]
            )
        )
        
        center3 = [0, -120]
        net.add_lane(
            "f",
            "g",
            CircularLane(
                center3,
                radii1,
                np.deg2rad(-90),
                np.deg2rad(-180),
                width=5,
                clockwise=False,
                line_types=lane1
            )
        )
        net.add_lane(
            "f",
            "g",
            CircularLane(
                center3,
                radii1 + 5,
                np.deg2rad(-90),
                np.deg2rad(-180),
                width=5,
                clockwise=False,
                line_types=lane2
            )
        )
        
        
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [-20, -120],
                [-20, -70],
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [-25, -120],
                [-25, -70],
                width=5,
                line_types=lane2
            )
        )
        
        center4 = [0, -70]
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii1,
                np.deg2rad(180),
                np.deg2rad(0),
                width=5,
                clockwise=False,
                line_types=lane1
            )
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii1 + 5,
                np.deg2rad(180),
                np.deg2rad(0),
                width=5,
                clockwise=False,
                line_types=lane2
            )
        )
        
        center5 = [35, -70]
        radii2 = 10
        net.add_lane(
            "i",
            "j",
            CircularLane(
                center5,
                radii2 + 5,
                np.deg2rad(-180),
                np.deg2rad(0),
                width=5,
                clockwise=True,
                line_types=lane1
            )
        )
        net.add_lane(
            "i",
            "j",
            CircularLane(
                center5,
                radii2,
                np.deg2rad(-180),
                np.deg2rad(0),
                width=5,
                clockwise=True,
                line_types=lane2
            )
        )
        
        center6 = [60, -70]
        net.add_lane(
            "j",
            "k",
            CircularLane(
                center6,
                radii2,
                np.deg2rad(180),
                np.deg2rad(0),
                clockwise=False,
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "j",
            "k",
            CircularLane(
                center6,
                radii2 + 5,
                np.deg2rad(180),
                np.deg2rad(0),
                clockwise=False,
                width=5,
                line_types=lane2
            )
        )
        
        net.add_lane(
            "k",
            "l",
            StraightLane(
                [70, -70],
                [70, -90],
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "k",
            "l",
            StraightLane(
                [75, -70],
                [75, -90],
                width=5,
                line_types=lane2
            )
        )
        
        center7 = [60, -90]
        net.add_lane(
            "l",
            "m",
            CircularLane(
                center7,
                radii2,
                np.deg2rad(0),
                np.deg2rad(-90),
                clockwise=False,
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "l",
            "m",
            CircularLane(
                center7,
                radii2 + 5,
                np.deg2rad(0),
                np.deg2rad(-90),
                clockwise=False,
                width=5,
                line_types=lane2
            )
        )
        
        net.add_lane(
            "m",
            "n",
            StraightLane(
                [60, -100],
                [20, -100],
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "m",
            "n",
            StraightLane(
                [60, -105],
                [20, -105],
                width=5,
                line_types=lane2
            )
        )
        
        center8 = [20, -90]
        net.add_lane(
            "n",
            "o",
            CircularLane(
                center8,
                radii2,
                np.deg2rad(-90),
                np.deg2rad(-180),
                clockwise=False,
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "n",
            "o",
            CircularLane(
                center8,
                radii2 + 5,
                np.deg2rad(-90),
                np.deg2rad(-180),
                clockwise=False,
                width=5,
                line_types=lane2
            )
        )
        
        net.add_lane(
            "o",
            "p",
            StraightLane(
                [10, -90],
                [10, -70],
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "o",
            "p",
            StraightLane(
                [5, -90],
                [5, -70],
                width=5,
                line_types=lane2
            )
        )
        
        center9 = [0, -70]
        radii3 = 5
        net.add_lane(
            "p",
            "q",
            CircularLane(
                center9,
                radii3 + 5,
                np.deg2rad(0),
                np.deg2rad(180),
                clockwise=True,
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "p",
            "q",
            CircularLane(
                center9,
                radii3,
                np.deg2rad(0),
                np.deg2rad(180),
                clockwise=True,
                width=5,
                line_types=lane2
            )
        )
        
        net.add_lane(
            "q",
            "r",
            StraightLane(
                [-10, -70],
                [-10, -90],
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "q",
            "r",
            StraightLane(
                [-5, -70],
                [-5, -90],
                width=5,
                line_types=lane2
            )
        )
        
        center10 = [25, -90]
        radii4 = 30
        net.add_lane(
            "r",
            "s",
            CircularLane(
                center10,
                radii4 + 5,
                np.deg2rad(-180),
                np.deg2rad(-90),
                clockwise=True,
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "r",
            "s",
            CircularLane(
                center10,
                radii4,
                np.deg2rad(-180),
                np.deg2rad(-90),
                clockwise=True,
                width=5,
                line_types=lane2
            )
        )
        
        net.add_lane(
            "s",
            "t",
            StraightLane(
                [25, -125],
                [70, -125],
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "s",
            "t",
            StraightLane(
                [25, -120],
                [70, -120],
                width=5,
                line_types=lane2
            )
        )
        
        center11 = [70, -90]
        net.add_lane(
            "t",
            "u",
            CircularLane(
                center11,
                radii4 + 5,
                np.deg2rad(-90),
                np.deg2rad(0),
                clockwise=True,
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "t",
            "u",
            CircularLane(
                center11,
                radii4,
                np.deg2rad(-90),
                np.deg2rad(0),
                clockwise=True,
                width=5,
                line_types=lane2
            )
        )
        
        
        center12 = [45, -90]
        radii5 = 55
        net.add_lane(
            "u",
            "v",
            CircularLane(
                center12,
                radii5 + 5,
                np.deg2rad(0),
                np.deg2rad(90),
                clockwise=True,
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "u",
            "v",
            CircularLane(
                center12,
                radii5,
                np.deg2rad(0),
                np.deg2rad(90),
                clockwise=True,
                width=5,
                line_types=lane2
            )
        )
        
        net.add_lane(
            "v",
            "w",
            StraightLane(
                [45, -30],
                [0, -30],
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "v",
            "w",
            StraightLane(
                [45, -35],
                [0, -35],
                width=5,
                line_types=lane2
            )
        )
        
        center13 = [0, -15]
        radii6 = 15
        
        net.add_lane(
            "w",
            "a",
            CircularLane(
                center13,
                radii6,
                np.deg2rad(-90),
                np.deg2rad(-270),
                clockwise=False,
                width=5,
                line_types=lane1
            )
        )
        net.add_lane(
            "w",
            "a",
            CircularLane(
                center13,
                radii6 + 5,
                np.deg2rad(-90),
                np.deg2rad(-270),
                clockwise=False,
                width=5,
                line_types=lane2
            )
        )
        
        
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road
        
    def _make_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        
        # Since only 1 controlle vehicle this will have the size of '1'
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            print("I am printing how many times?")
            vehicle = Vehicle.create_random(
                self.road,
                speed=5,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            
            # This creates an MDP vehicle whic
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            
            
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)


            # This creates the uncontrolled vehicles
            for _ in range(others):
                print("\thow many times am I printing here?")
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                
                
                
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