import numpy as np
from typing import Tuple
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs import HighwayEnv, StraightLane, CircularLane
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle


class ExitEnv(HighwayEnv):
    """
    """
    HIGH_SPEED_REWARD = 0.1

    def default_config(self) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "collision_reward": 0,
            "goal_reward": 1,
            "vehicles_count": 5,
            "controlled_vehicles": 1,
            "duration": 15,  # [s],
            "scaling": 5
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self, road_length=1000, exit_position=300, exit_length=100) -> None:
        net = RoadNetwork.straight_road_network(self.config["lanes_count"], start=0,
                                                length=exit_position, nodes_str=("0", "1"))
        net = RoadNetwork.straight_road_network(self.config["lanes_count"] + 1, start=exit_position,
                                                length=exit_length, nodes_str=("1", "2"), net=net)
        net = RoadNetwork.straight_road_network(self.config["lanes_count"], start=exit_position+exit_length,
                                                length=road_length-exit_position-exit_length,
                                                nodes_str=("2", "3"), net=net)
        exit_position = np.array([exit_position + exit_length, self.config["lanes_count"] * CircularLane.DEFAULT_WIDTH])
        radius = 150
        exit_center = exit_position + np.array([0, radius])
        lane = CircularLane(center=exit_center,
                            radius=radius,
                            start_phase=3*np.pi/2,
                            end_phase=2*np.pi,
                            forbidden=True)
        net.add_lane("2", "exit", lane)

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for _ in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class.create_random(self.road,
                                                                   speed=25,
                                                                   lane_from="0",
                                                                   lane_to="1",
                                                                   lane_id=0,
                                                                   spacing=self.config["ego_spacing"])
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            lane_id = self.road.np_random.randint(low=1, high=self.config["lanes_count"])
            vehicle = vehicles_type.create_random(self.road,
                                                  lane_from="0",
                                                  lane_to="1",
                                                  lane_id=lane_id,
                                                  spacing=1 / self.config["vehicles_density"],
                                                  ).plan_route_to("3")
            vehicle.enable_lane_change = False
            self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        lane_index = self.vehicle.target_lane_index if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index
        goal_reached = lane_index == ("1", "2", self.config["lanes_count"]) or lane_index == ("2", "exit", 0)
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = self.config["collision_reward"] * self.vehicle.crashed \
                 + self.config["goal_reward"] * goal_reached \
                 + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)

        reward = utils.lmap(reward,
                          [self.config["collision_reward"], self.config["goal_reward"] + self.HIGH_SPEED_REWARD],
                          [0, 1])
        print(lane_index, goal_reached, reward)
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or self.steps >= self.config["duration"]


register(
    id='exit-v0',
    entry_point='highway_env.envs:ExitEnv',
)
