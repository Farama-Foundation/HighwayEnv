from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import ConnectedLaneNeighboursMixin
from highway_env.envs.common.action import Action
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.lane import CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class ExitEnv(HighwayEnv):
    """ """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        utils.update_config(
            config,
            {
                "observation": {
                    "type": "ExitObservation",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "clip": False,
                },
                "action": {"type": "DiscreteMetaAction", "target_speeds": [18, 24, 30]},
                "lanes_count": 6,
                "collision_reward": 0,
                "high_speed_reward": 0.1,
                "right_lane_reward": 0,
                "normalize_reward": True,
                "goal_reward": 1,
                "vehicles_count": 20,
                "vehicles_density": 1.5,
                "controlled_vehicles": 1,
                "duration": 18,  # [s],
                "simulation_frequency": 5,
                "scaling": 5,
            },
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        info.update({"is_success": self._is_success()})
        return obs, reward, terminated, truncated, info

    def _create_road(
        self, road_length=1000, exit_position=400, exit_length=100
    ) -> None:
        net = RoadNetwork.straight_road_network(
            self.config["lanes_count"],
            start=0,
            length=exit_position,
            nodes_str=("0", "1"),
        )
        net = RoadNetwork.straight_road_network(
            self.config["lanes_count"] + 1,
            start=exit_position,
            length=exit_length,
            nodes_str=("1", "2"),
            net=net,
        )
        net = RoadNetwork.straight_road_network(
            self.config["lanes_count"],
            start=exit_position + exit_length,
            length=road_length - exit_position - exit_length,
            nodes_str=("2", "3"),
            net=net,
        )
        for _from in net.graph:
            for _to in net.graph[_from]:
                for _id in range(len(net.graph[_from][_to])):
                    net.get_lane((_from, _to, _id)).speed_limit = 26 - 3.4 * _id
        exit_position = np.array(
            [
                exit_position + exit_length,
                self.config["lanes_count"] * CircularLane.DEFAULT_WIDTH,
            ]
        )
        radius = 150
        exit_center = exit_position + np.array([0, radius])
        lane = CircularLane(
            center=exit_center,
            radius=radius,
            start_phase=3 * np.pi / 2,
            end_phase=2 * np.pi,
            forbidden=True,
        )
        net.add_lane("2", "exit", lane)

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
            neighbour_vehicles_connected_lanes=self.config[
                "neighbour_vehicles_connected_lanes"
            ],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for _ in range(self.config["controlled_vehicles"]):
            vehicle = Vehicle.create_random(
                self.road,
                speed=25.0,
                lane_from="0",
                lane_to="1",
                lane_id=0,
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            lanes = np.arange(self.config["lanes_count"])
            lane_id = self.road.np_random.choice(
                lanes, size=1, p=lanes / lanes.sum()
            ).astype(int)[0]
            lane = self.road.network.get_lane(("0", "1", lane_id))
            vehicle = vehicles_type.create_random(
                self.road,
                lane_from="0",
                lane_to="1",
                lane_id=lane_id,
                speed=lane.speed_limit,
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
        reward = sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["goal_reward"]],
                [0, 1],
            )
            reward = np.clip(reward, 0, 1)
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        lane_index = (
            self.vehicle.target_lane_index
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index
        )
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": self.vehicle.crashed,
            "goal_reward": self._is_success(),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "right_lane_reward": lane_index[-1],
        }

    def _is_success(self):
        lane_index = (
            self.vehicle.target_lane_index
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index
        )
        goal_reached = lane_index == (
            "1",
            "2",
            self.config["lanes_count"],
        ) or lane_index == ("2", "exit", 0)
        return goal_reached

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class ConnectedLaneExitEnv(ConnectedLaneNeighboursMixin, ExitEnv):
    pass


class ExitEnvV2(ConnectedLaneExitEnv):
    """
    Exit environment with a reward that depends on actually taking the exit.

    The episode ends when the ego-vehicle takes the exit ramp, crashes, or drives past the exit.
    Taking the exit is rewarded, crashing and missing the exit are penalised, and a small shaping
    term rewards lane changes towards the exit lane (and takes it back for lane changes away from it).
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        utils.update_config(
            config,
            {
                "collision_reward": -1,
                "goal_reward": 1,
                "missed_exit_reward": -1,
                "high_speed_reward": 0.02,
                "lane_progress_reward": 0.5,
                "normalize_reward": False,
            },
        )
        return config

    def _reset(self) -> None:
        super()._reset()
        self._previous_lane_progress = self._lane_progress()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._previous_lane_progress = self._lane_progress()
        return super().step(action)

    def _lane_progress(self) -> float:
        """Position of the ego-vehicle's lane between the leftmost lane (0) and the exit lane (1)."""
        _from, _to, lane_id = self.vehicle.lane_index
        if (_from, _to) == ("2", "exit"):
            return 1.0
        return lane_id / self.config["lanes_count"]

    def _rewards(self, action: Action) -> dict[str, float]:
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        missed_exit = not self.vehicle.crashed and (
            self._has_passed_exit() or (self._is_truncated() and not self._is_success())
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "goal_reward": float(self._is_success()),
            "missed_exit_reward": float(missed_exit),
            "high_speed_reward": float(np.clip(scaled_speed, 0, 1)),
            "lane_progress_reward": self._lane_progress()
            - self._previous_lane_progress,
        }

    def _is_success(self) -> bool:
        """The ego-vehicle has left the highway on the exit ramp."""
        return self.vehicle.lane_index[:2] == ("2", "exit")

    def _has_passed_exit(self) -> bool:
        """The ego-vehicle has driven past the exit on the highway."""
        return self.vehicle.lane_index[:2] == ("2", "3")

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed, took the exit, or drove past it."""
        return self.vehicle.crashed or self._is_success() or self._has_passed_exit()


# class DenseLidarExitEnv(DenseExitEnv):
#     @classmethod
#     def default_config(cls) -> dict:
#         return dict(super().default_config(),
#                     observation=dict(type="LidarObservation"))
