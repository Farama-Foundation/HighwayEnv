from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
# from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
# from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle

Observation = np.ndarray


class AEBEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 1,
            "duration": 15,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "normalize_reward": True,
            "offroad_terminal": False,
            "simulation_frequency": 5,
            "centering_position": [0.7, 0.5],
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=70),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        self.controlled_vehicles = []
        agent_vehicle = ControlledVehicle(
            self.road,
            position=(100, 0),
            speed=30,
        )
        self.controlled_vehicles.append(agent_vehicle)
        self.road.vehicles.append(agent_vehicle)
        subject_vehicle = IDMVehicle(
            self.road,
            position=(70, 0),
            speed=32,
        )
        self.road.vehicles.append(subject_vehicle)
        # """Create some new random vehicles of a given type, and add them on the road."""
        # other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        # self.controlled_vehicles = []
        # for others in other_per_controlled:
        #     vehicle = Vehicle.create_random(
        #         self.road,
        #         speed=25,
        #         lane_id=self.config["initial_lane_id"],
        #         spacing=self.config["ego_spacing"]
        #     )
        #     vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
        #     self.controlled_vehicles.append(vehicle)
        #     self.road.vehicles.append(vehicle)

        #     for _ in range(others):
        #         vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
        #         vehicle.randomize_behavior()
        #         self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # rewards = self._rewards(action)
        # reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        # if self.config["normalize_reward"]:
        #     reward = utils.lmap(reward,
        #                         [self.config["collision_reward"],
        #                          self.config["high_speed_reward"] + self.config["right_lane_reward"]],
        #                         [0, 1])
        # reward *= rewards['on_road_reward']
        # return reward
        return 0

    def _rewards(self, action: Action) -> Dict[Text, float]:
        # neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        # lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
        #     else self.vehicle.lane_index[2]
        # # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        # forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        # scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        # return {
        #     "collision_reward": float(self.vehicle.crashed),
        #     "right_lane_reward": lane / max(len(neighbours) - 1, 1),
        #     "high_speed_reward": np.clip(scaled_speed, 0, 1),
        #     "on_road_reward": float(self.vehicle.on_road)
        # }
        return {}

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
