from typing import Dict, Text, Tuple, Optional

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
# from highway_env.utils import near_split
from highway_env.vehicle.controller_aeb import AEBControlledVehicle
from highway_env.vehicle.controller_aeb_ncap import AEBNCAPControlledVehicle
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
            "longi_aggr": True,
            "reward_min": -1.0,
            "reward_max": 1.0,
            "reward_adversarial": False,
            "reward_collision": False,
            "init_state": None,
            "ncap": False,
            "speed_limit": 30,
        })
        return config
    
    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        super().__init__(config, render_mode)
        self.headway = 0

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=70),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        if self.config["init_state"] is None:
            init_speed_range = [25.0, 35.0]
            init_x_range = [15, 50.0] # vehicle length: 5 [m]
            
            agent_init_x = self.np_random.random() * (init_x_range[1] - init_x_range[0]) + init_x_range[0]
            agent_init_spd = self.np_random.random() * (init_speed_range[1] - init_speed_range[0]) + init_speed_range[0]
            subject_init_spd = self.np_random.random() * (init_speed_range[1] - init_speed_range[0]) + init_speed_range[0]
            
            # while True:
            #     spd_diff = subject_init_spd - agent_init_spd
            #     t = spd_diff / 6.0
            #     if spd_diff * 0.5 * t < agent_init_x:
            #         break
                
            #     agent_init_x = np.random.sample() * (init_x_range[1] - init_x_range[0]) + init_x_range[0]
            #     agent_init_spd = np.random.sample() * (init_speed_range[1] - init_speed_range[0]) + init_speed_range[0]
            #     subject_init_spd = np.random.sample() * (init_speed_range[1] - init_speed_range[0]) + init_speed_range[0]
            
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
                target_speed=self.config["speed_limit"],
                longi_aggr=self.config["longi_aggr"],
            )
            self.road.vehicles.append(subject_vehicle)
            # print(f'initial condition *** agent: pos - {agent_init_x}, spd - {agent_init_spd}; subject: spd - {subject_init_spd}')
        else:
            init_state = self.config["init_state"] # [dhw, sv_v, agent_v, agent_a if ncap else sv_target_v]
            agent_init_x = init_state[0] + 5.0
            agent_init_spd = init_state[2]
            subject_init_spd = init_state[1]
            subject_target_spd = init_state[3]
            
            self.controlled_vehicles = []
            if not self.config["ncap"]:
                agent_vehicle = AEBControlledVehicle(
                    self.road,
                    position=(agent_init_x, 0),
                    speed=agent_init_spd,
                    target_speed=agent_init_spd,
                )
            else:
                agent_vehicle = AEBNCAPControlledVehicle(
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
                target_speed=self.config["speed_limit"],
                longi_aggr=self.config["longi_aggr"],
            )
            self.road.vehicles.append(subject_vehicle)
        
    def _reward_default(self) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        r_max = self.config["reward_max"]
        r_min = self.config["reward_min"]
        
        def normalize(r, r_min, r_max):
            r = (r - r_min) / (r_max - r_min)
            return np.clip(r, 0, 1.0)
        
        if self._is_terminated():
            reward = r_min
            self.headway = 0.0
        else:
            nrd = 25.0    # no reward distance threshold [m]
            safe_margin = 1.0 # [m]
            safe_headway = 0.5
            agent_vehicle = self.road.vehicles[0]
            subject_vehicle = self.road.vehicles[1]
            
            headway = agent_vehicle.position[0] - subject_vehicle.position[0] - agent_vehicle.LENGTH
            self.headway = headway
            
            if headway < safe_headway:
                reward = r_min
            elif headway < safe_headway + safe_margin:
                reward = (r_max - r_min) / safe_margin * headway + r_min - safe_headway / safe_margin * (r_max - r_min)
            else:
                reward = max(0.0, r_max * (headway - nrd) / (safe_headway + safe_margin - nrd))
        return normalize(reward, r_min, r_max)
    
    def _reward(self, action: Action) -> float:
        
        def normalize(r, r_min, r_max):
            r = (r - r_min) / (r_max - r_min)
            return np.clip(r, 0, 1.0)
        
        if self.config["reward_adversarial"] and self.config["reward_collision"]:
            # raise NotImplementedError
            reward = normalize(self._reward_adversarial() + (self._reward_collision() - 1.0) * 2.0, -1.0, 1.0)
            return reward
        elif self.config["reward_adversarial"]:
            return self._reward_adversarial()
        elif self.config["reward_collision"]:
            return self._reward_collision()
        else:
            return self._reward_default()
        
    def _reward_collision(self) -> float:
        rew_collison = 0.0
        rew_nocollision = 1.0
        if self._is_terminated():
            return rew_collison
        else:
            return rew_nocollision
        
    def _reward_adversarial(self) -> float:
        # rew_max = 1.0
        # rew_min = 0.0
        # no_reward_dist = 5.0 # [m]
        
        # agent_vehicle = self.road.vehicles[0]
        # subject_vehicle = self.road.vehicles[1]
        
        # headway = agent_vehicle.position[0] - subject_vehicle.position[0] - agent_vehicle.LENGTH
        # self.headway = headway
        
        # reward = max(rew_min, min(rew_max, -rew_max / no_reward_dist * headway + rew_max))
        # return reward
        
        def normalize(r, r_min, r_max):
            r = (r - r_min) / (r_max - r_min)
            return np.clip(r, 0, 1.0)
        
        r_min = 0.0
        r_max = 1.0
        nrd = 25.0    # no reward distance threshold [m]
        safe_margin = 1.0 # [m]
        safe_headway = 0.5
        agent_vehicle = self.road.vehicles[0]
        subject_vehicle = self.road.vehicles[1]
        
        headway = agent_vehicle.position[0] - subject_vehicle.position[0] - agent_vehicle.LENGTH
        self.headway = headway
        
        if headway < safe_headway:
            reward = r_min
        elif headway < safe_headway + safe_margin:
            reward = (r_max - r_min) / safe_margin * headway + r_min - safe_headway / safe_margin * (r_max - r_min)
        else:
            reward = max(0.0, r_max * (headway - nrd) / (safe_headway + safe_margin - nrd))
        return normalize(reward, r_min, r_max)

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
    
