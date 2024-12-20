from __future__ import annotations

import math

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.network_builder import CircularPath, NetworkBuilder, Path, StraightPath
from highway_env.road.lanes.abstract_lanes import AbstractLane
from highway_env.road.lanes.lane_utils import LineType
from highway_env.road.lanes.unweighted_lanes import CircularLane, SineLane, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import Road, RoadNetwork, WeightedRoadnetwork
from highway_env.utils import Vector, near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class Template(AbstractEnv):
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
                    "target_speeds": [-10, 0, 5, 10, 20],  # np.linspace(0, 30, 10)
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

        net.add_lane(
            "ser", "R-1:s-in", StraightLane([2, 170], [2, 42.5], line_types=(s, c))
        )
        net.add_lane(
            "R-1:s-out", "sxr", StraightLane([-2, 42.5], [-2, 170], line_types=(n, c))
        )

        nb.add_roundabout(
            "R-1",
            {
                nb.CardinalDirection.NORTH: [-2, -42.5],
                nb.CardinalDirection.SOUTH: [2, 42.5],
                nb.CardinalDirection.EAST: [42.5, -2],
                nb.CardinalDirection.WEST: [-42.5, 2],
            },
        )

        net.add_lane("v", "hj", StraightLane([0, -24], [0, 24], 1, line_types=(s, c)))
        net.add_lane(
            "alsjd", "adjfa", StraightLane([-24, 0], [24, 0], 1, line_types=(s, c))
        )

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

        ego_lane = self.road.network.get_lane(
            ("ser", "R-1:s-in", 0)
        )  # This is to place the car on a road between two points
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(120, 0),  # Use the first value to place car down the road
            speed=0,  # Speed of car
            heading=ego_lane.heading_at(
                90
            ),  # Use this to change the direction of the car. "0" is north
        )

        ego_vehicle.plan_route_to("sxr")

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

