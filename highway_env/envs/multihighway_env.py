import numpy as np
from typing import Tuple
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle, ControlledVehicle



import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle



class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    RIGHT_LANE_REWARD: float = 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    HIGH_SPEED_REWARD: float = 0.4
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds."""

    LANE_CHANGE_REWARD: float = 0
    """The reward received at each lane change action."""
    # ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}
    @classmethod
    def default_config(self) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_vehicle_count": 10,
            "duration": 40,  # [s]
            "initial_spacing": 2,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for _ in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class.create_random(self.road, 25, spacing=self.config["initial_spacing"])
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            self.road.vehicles.append(vehicles_type.create_random(self.road))

    # def _reward(self, action: Action) -> float:
    #     """
    #     The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
    #     :param action: the last action performed
    #     :return: the corresponding reward
    #     """
    #     neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
    #     lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
    #         else self.vehicle.lane_index[2]
    #     speed = self.vehicle.speed_index if isinstance(self.vehicle, MDPVehicle) \
    #         else MDPVehicle.speed_to_index(self.vehicle.speed)
    #     reward = \
    #         + self.config["collision_reward"] * self.vehicle.crashed \
    #         + self.RIGHT_LANE_REWARD * lane / (len(neighbours) - 1) \
    #         + self.HIGH_SPEED_REWARD * speed / (MDPVehicle.SPEED_COUNT - 1)
    #     reward = utils.lmap(reward,
    #                       [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
    #                       [0, 1])
    #     reward = 0 if not self.vehicle.on_road else reward
    #     return reward
    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        speed = self.vehicle.speed_index if isinstance(self.vehicle, MDPVehicle) \
            else MDPVehicle.speed_to_index(self.vehicle.speed)
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * lane / (len(neighbours) - 1) \
            + self.HIGH_SPEED_REWARD * speed / (MDPVehicle.SPEED_COUNT - 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward





    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)


    # def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
    #     reward = self.config["collision_reward"] * vehicle.crashed \
    #              + self.HIGH_SPEED_REWARD * (vehicle.speed_index == vehicle.SPEED_COUNT - 1)
    #     reward = self.ARRIVED_REWARD if self.has_arrived(vehicle) else reward
    #     if self.config["normalize_reward"]:
    #         reward = utils.lmap(reward, [self.config["collision_reward"], self.ARRIVED_REWARD], [0, 1])
    #     return reward


    def _is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
            or self.steps >= self.config["duration"]* self.config["policy_frequency"]or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)


    # def _is_terminal(self) -> bool:
    #     return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
    #            or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) \
    #            or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)



class MultiAgentHighwayEnv(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                 "type": "MultiAgentAction",
                 "action_config": {
                     "type": "DiscreteMetaAction"
                 }
            },
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }
            },
            "controlled_vehicles": 2
        })
        return config


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-multi-agent-v0',
    entry_point='highway_env.envs:MultiAgentHighwayEnv',
)