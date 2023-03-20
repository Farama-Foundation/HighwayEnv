from typing import Dict, Text, Tuple

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
# from highway_env.utils import near_split
from highway_env.vehicle.controller_aeb import AEBControlledVehicle
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
                "type": "AEB"
            },
            "action": {
                "type": "AEBAction",
                "longitudinal": True,
                "lateral": False,
            },
            "lanes_count": 1,
            "duration": 15,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "normalize_reward": True,
            "offroad_terminal": False,
            "simulation_frequency": 15,
            "policy_frequency": 1,
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
        init_speed_range = [25.0, 35.0]
        init_x_range = [15.0, 50.0]
        
        agent_init_x = np.random.sample() * (init_x_range[1] - init_x_range[0]) + init_x_range[0]
        agent_init_spd = np.random.sample() * (init_speed_range[1] - init_speed_range[0]) + init_speed_range[0]
        subject_init_spd = np.random.sample() * (init_speed_range[1] - init_speed_range[0]) + init_speed_range[0]
        
        while True:
            spd_diff = subject_init_spd - agent_init_spd
            t = spd_diff / 6.0
            if spd_diff * 0.5 * t < agent_init_x:
                break
            
            agent_init_x = np.random.sample() * (init_x_range[1] - init_x_range[0]) + init_x_range[0]
            agent_init_spd = np.random.sample() * (init_speed_range[1] - init_speed_range[0]) + init_speed_range[0]
            subject_init_spd = np.random.sample() * (init_speed_range[1] - init_speed_range[0]) + init_speed_range[0]
        
        self.controlled_vehicles = []
        agent_vehicle = AEBControlledVehicle(
            self.road,
            position=(agent_init_x, 0),
            speed=agent_init_spd,
            target_speed=agent_init_spd,
        )
        self.controlled_vehicles.append(agent_vehicle)
        self.road.vehicles.append(agent_vehicle)
        subject_vehicle = IDMVehicle(
            self.road,
            position=(0, 0),
            speed=subject_init_spd,
            target_speed=subject_init_spd,
        )
        self.road.vehicles.append(subject_vehicle)
        # print(f'initial condition *** agent: pos - {agent_init_x}, spd - {agent_init_spd}; subject: spd - {subject_init_spd}')

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        r_max = 1.0
        r_min = -1.0
        
        def normalize(r, r_min, r_max):
            r = (r - r_min) / (r_max - r_min)
            return np.clip(r, 0, 1.0)
        
        if self._is_terminated():
            reward = r_min
        else:
            nrd = 25.0    # no reward distance threshold [m]
            safe_margin = 1.0 # [m]
            safe_headway = 0.5
            agent_vehicle = self.road.vehicles[0]
            subject_vehicle = self.road.vehicles[1]
            
            headway = agent_vehicle.position[0] - subject_vehicle.position[0] - agent_vehicle.LENGTH
            
            if headway < safe_headway:
                reward = r_min
            elif headway < safe_headway + safe_margin:
                reward = (r_max - r_min) / safe_margin * headway + r_min - safe_headway / safe_margin * (r_max - r_min)
            else:
                reward = max(0.0, r_max * (headway - nrd) / (safe_headway + safe_margin - nrd))
        return normalize(reward, r_min, r_max)

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
