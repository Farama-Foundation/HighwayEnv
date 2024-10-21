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


class HomemadeHighway(AbstractEnv):
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

        net = RoadNetwork()
        line_none, line_continuous, line_striped = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        turn_radius = 20
        left_turn = False
        right_turn = True
        lane_width = StraightLane.DEFAULT_WIDTH
        
        line_type = [[line_continuous, line_striped], [line_none, line_continuous]]
        line_type_merge = [[line_continuous, line_striped], [line_none, line_striped]]
        lane_placement = [0, lane_width]
        
        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        
        for lane in range(2):
            
            """First highway circle"""
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0, 0 + lane_placement[lane]],
                    [300, 0 + lane_placement[lane]],
                    lane_width,
                    line_type[lane]
                )
            )
            
            # Merge section
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [300, 0 + lane_placement[lane]],
                    [600, 0 + lane_placement[lane]],
                    lane_width,
                    line_type_merge[lane]
                )
            )
            
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [600, 0 + lane_placement[lane]],
                    [700, 0 + lane_placement[lane]],
                    lane_width,
                    line_type[lane]
                )
            )
            
            center1 = [700, -20]
            net.add_lane(
                "d",
                "e",
                CircularLane(
                    center1,
                    turn_radius + lane_placement[lane],
                    np.deg2rad(90),
                    np.deg2rad(0),
                    left_turn,
                    lane_width,
                    line_type[lane]
                )
            )
            
            net.add_lane(
                "e",
                "f",
                StraightLane(
                    [720 + lane_placement[lane], -20],
                    [720 + lane_placement[lane], -120],
                    lane_width,
                    line_type[lane]
                )
            )
            
            center2 = [700, -120]
            net.add_lane(
                "f",
                "g",
                CircularLane(
                    center2,
                    turn_radius + lane_placement[lane],
                    np.deg2rad(0),
                    np.deg2rad(-90),
                    left_turn,
                    lane_width,
                    line_type[lane]
                )
            )
            
            net.add_lane(
                "g",
                "h",
                StraightLane(
                    [700, -140 - lane_placement[lane]],
                    [0, -140 - lane_placement[lane]],
                    lane_width,
                    line_type[lane]
                )
            )
            
            center3 = [0, -120]
            net.add_lane(
                "h",
                "i",
                CircularLane(
                    center3,
                    turn_radius + lane_placement[lane],
                    np.deg2rad(-90),
                    np.deg2rad(-180),
                    left_turn,
                    lane_width,
                    line_type[lane]
                )
            )
            
            net.add_lane(
                "i",
                "j",
                StraightLane(
                    [-20 - lane_placement[lane], -120],
                    [-20 - lane_placement[lane], -20],
                    lane_width,
                    line_type[lane]
                )
            )
            
            center4 = [0, -20]
            net.add_lane(
                "j",
                "a",
                CircularLane(
                    center4,
                    turn_radius + lane_placement[lane],
                    np.deg2rad(180),
                    np.deg2rad(90),
                    left_turn,
                    lane_width,
                    line_type[lane]
                )
            )


        for lane in range(2):
            """Second highway circle"""
            
        
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [300, 2 * lane_width + lane_placement[lane]],
                    [600, 2 * lane_width + lane_placement[lane]],
                    lane_width,
                    [
                        line_none,
                        line_striped if lane == 0 else line_continuous
                    ]
                )
            )
            
            center5 = [600, 3 * lane_width + turn_radius]
            net.add_lane(
                "c",
                "k",
                CircularLane(
                    center5,
                    turn_radius + lane_width - lane_placement[lane],
                    np.deg2rad(-90),
                    np.deg2rad(0),
                    right_turn,
                    lane_width,
                    line_type[lane]
                )
            )
            
            net.add_lane(
                "k",
                "l",
                StraightLane(
                    [620 + lane_width - lane_placement[lane], 3 * lane_width + turn_radius],
                    [620 + lane_width - lane_placement[lane], 3 * lane_width + turn_radius + 100],
                    lane_width,
                    line_type[lane]
                )
            )

            center6 = [600, 3 * lane_width + turn_radius + 100]
            net.add_lane(
                "l",
                "m",
                CircularLane(
                    center6,
                    turn_radius + lane_width - lane_placement[lane],
                    np.deg2rad(0),
                    np.deg2rad(90),
                    right_turn,
                    lane_width,
                    line_type[lane]
                )
            )
            
            net.add_lane(
                "m",
                "n",
                StraightLane(
                    [600, 4 * lane_width + 2 * turn_radius + 100 - lane_placement[lane]],
                    [300, 4 * lane_width + 2 * turn_radius + 100 - lane_placement[lane]],
                    lane_width,
                    line_type[lane]
                )
            )
            
            center7 = [300, 3 * lane_width + turn_radius + 100]
            net.add_lane(
                "n",
                "o",
                CircularLane(
                    center7,
                    turn_radius + lane_width - lane_placement[lane],
                    np.deg2rad(90),
                    np.deg2rad(180),
                    right_turn,
                    lane_width,
                    line_type[lane]
                )
            )
            
            net.add_lane(
                "o",
                "p",
                StraightLane(
                    [280 - lane_width + lane_placement[lane], 3 * lane_width + turn_radius + 100],
                    [280 - lane_width + lane_placement[lane], 2 * lane_width + turn_radius ],
                    lane_width,
                    line_type[lane]
                )
            )
            
            center8 = [300, 3 * lane_width + turn_radius]
            net.add_lane(
                "p",
                "b",
                CircularLane(
                    center8,
                    turn_radius + lane_width - lane_placement[lane],
                    np.deg2rad(-180),
                    np.deg2rad(-90),
                    right_turn,
                    lane_width,
                    line_type[lane]
                )
            )
            
        
        road = Road(
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

        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 0)).position(0, 0), speed=20
        )
        ego_vehicle.target_speed = 30
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