from typing import Dict, Text

import numpy as np
import gymnasium as gym

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action, action_factory
from highway_env.envs.common.agents import AttackerAgent, VictimAgent
# from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.graphics import EnvViewer
from gymnasium import spaces
from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text
import torch
import random
import time

Observation = np.ndarray


class HighwayEnvCentralizedPenalty(AbstractEnv):
    victim = None
    victim_action = None
    r_sum = 0
    c_sum = 0
    
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    # def __init__(self, attacker_num, attacker_net_cls, victim_net_cls, victim_net_path, victim_predict_fn, device, config: dict = None, render_mode: Optional[str] = None, *victim_net_args) -> None:
    #     super().__init__(config, render_mode)
    #     self.victim_net = victim_net_cls(*victim_net_args).to(device)
    #     self.victim_net.load_state_dict(torch.load(victim_net_path))
    #     self.victim_predict_fn = victim_predict_fn
    #     self.attacker_agent_set = set()
    #     for i in range(attacker_num):
    #         self.attacker_agent_set.add(AttackerAgent(attacker_net_cls, i))
    def load_agents(self, attacker_num, victim_agent: VictimAgent):
        self.victim_agent = victim_agent
        
        # self.attacker_agent_set = set()
        # for i in range(attacker_num):
        #     self.attacker_agent_set.add(AttackerAgent(i))

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        self.observation_type = observation_factory(self, self.config["observation"])
        self.victim_action_type = action_factory(self, self.config["victim_action"])
        self.victim_observation_type = observation_factory(self, self.config["victim_observation"])
        
        if self.victim:
            self.victim_observation_type.observer_vehicle = self.victim
            self.observation_type.observer_vehicle = self.victim
            self.victim_action_type.controlled_vehicle = self.victim
        self.observation_space = self.observation_type.space()
        # print("self.observation_space: {}".format(self.observation_space))
        # TODO: make 5 configurable
        self.action_space = spaces.MultiDiscrete([5 for _ in range(self.config["attacker_num"])])
        
        self.action_type = action_factory(self, self.config["action"])
        # self.action_space = self.action_type.space()

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "see_behind": True,
                "vehicles_count": 5
            },

            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                }
            },
            "victim_action": {
                "type": "DiscreteMetaAction",
            },
            "victim_observation": {
                    "type": "Kinematics",
            },
            "attacker_action": {
                "type": "DiscreteMetaAction",
            },
            "attacker_num": 4,
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 4,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -5,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            "randomize_starting_position": False,
            "time_penalty": -0.1,
            "attacker_collide_each_other_reward": -5,
            "vicitm_collision_reward": 10,
            "constraint_env": False,
            "close_vehicle_threshold": 12,
            "close_vehicle_cost": 5,
            "invalid_action_cost": 3,
            "diving_beside": 3,
            "vis": False
        })
        return config
    
    def _reset(self) -> None:
        self.r_sum = 0
        self.c_sum = 0
        self._create_road()
        self._create_vehicles()
        if self.viewer is not None:
            self.viewer.observer_vehicle = self.victim

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        # victim_index = (self.config["controlled_vehicles"]+1)//2
        if self.config["randomize_starting_position"] == False:
            self.victim_index = (self.config["controlled_vehicles"]+1)//2
            # self.victim_index = 0
        else:
            self.victim_index = random.randint(0, self.config["controlled_vehicles"])
        for i in range(self.config["controlled_vehicles"]+1):
            if i == self.victim_index:
                v = Vehicle.create_random(
                        self.road,
                        speed=25,
                        lane_id=self.config["initial_lane_id"],
                        spacing=self.config["ego_spacing"]
                    )
                self.victim = self.victim_action_type.vehicle_class(self.road, v.position, v.heading, v.speed)
                self.road.vehicles.append(self.victim)
                self.victim_observation_type.observer_vehicle = self.victim
                self.victim_action_type.controlled_vehicle = self.victim
            else:
                vehicle = Vehicle.create_random(
                    self.road,
                    speed=25,
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"]
                )
                vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)
            
    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        

        # TODO: calculate cost; we need to iterate every pair of vehicles
        if self.config["vis"]:
            print("action to take: ", action)
        if self.config["constraint_env"]:
            cost = self.calc_cost(action)
            self.c_sum += cost

        # # insert vulneralbility here
        # vehicle_ahead = vehicle_behind = vehicle_left = vehicle_right = False
        # victim_l_index = self.road.vehicles[self.victim_index].lane_index
        # for i in range(len(self.controlled_vehicles)):
        #     other_l_index = self.controlled_vehicles[i].lane_index
        #     if other_l_index[2] == victim_l_index[2]:
        #         # attacker is the same lane as the victim
        #         victim_local_x = self.road.network.get_lane(victim_l_index).local_coordinates(self.victim.position)[0]
        #         attacker_local_x = self.road.network.get_lane(victim_l_index).local_coordinates(self.controlled_vehicles[i].position)[0]
        #         if attacker_local_x > victim_local_x and (attacker_local_x - victim_local_x) <= self.config["close_vehicle_threshold"]:
        #             vehicle_ahead = True
        #         elif attacker_local_x < victim_local_x and (victim_local_x - attacker_local_x) <= self.config["close_vehicle_threshold"]:
        #             vehicle_behind = True
        #     elif other_l_index[2] < victim_l_index[2] and self.road.network.get_lane(other_l_index).is_reachable_from(self.victim.position):
        #         # attacker is on the left lane of the victim
        #         victim_local_x = self.road.network.get_lane(other_l_index).local_coordinates(self.victim.position)[0]
        #         attacker_local_x = self.road.network.get_lane(other_l_index).local_coordinates(self.controlled_vehicles[i].position)[0]
        #         if abs(victim_local_x - attacker_local_x) <= self.config["diving_beside"]:
        #             vehicle_left = True
        #     elif other_l_index[2] > victim_l_index[2] and self.road.network.get_lane(other_l_index).is_reachable_from(self.victim.position):
        #         # attacker is on the right lane of the victim
        #         victim_local_x = self.road.network.get_lane(other_l_index).local_coordinates(self.victim.position)[0]
        #         attacker_local_x = self.road.network.get_lane(other_l_index).local_coordinates(self.controlled_vehicles[i].position)[0]
        #         if abs(victim_local_x - attacker_local_x) <= self.config["diving_beside"]:
        #             vehicle_right = True
        # if vehicle_right and vehicle_left:
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     print("victim action overrided")
        #     self.victim_action = 0
        
        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)
        
        obs = self.observation_type.observe()
        victim_obs = self.victim_observation_type.observe()
        self.victim_action = self.victim_agent.select_action(victim_obs)
        if self.config["vis"]:
            print("victim action: {}".format(self.victim_action))
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(victim_obs, action, terminated)
        if self.render_mode == 'human':
            self.render()
        reward = reward - cost
        return obs, reward, terminated, truncated, info
    
    def calc_cost(self, action):
        victim_skipped = False
        j = 0
        cost = 0
        for i in range(len(self.road.vehicles)):
            if victim_skipped:
                j = i - 1
            else:
                j = i
            if self.road.vehicles[i] is self.victim:
                victim_skipped = True
                continue
            l_index = self.road.vehicles[i].lane_index[2]
            # print("lane index: ", l_index)
            if action[j] == 0:
                # lane left
                if l_index == 0:
                    # change lane is forbidden
                    if self.config["vis"]:
                        print("invalid action")
                    cost += self.config["invalid_action_cost"]
                    continue
                for k in range(len(self.road.vehicles)):
                    if k == i:
                        continue
                    other_l_index = self.road.vehicles[k].lane_index
                    if other_l_index[2] < l_index and self.road.network.get_lane(other_l_index).is_reachable_from(self.road.vehicles[i].position):
                        x = self.road.network.get_lane(other_l_index).local_coordinates(self.road.vehicles[i].position)[0]
                        other_x = self.road.network.get_lane(other_l_index).local_coordinates(self.road.vehicles[k].position)[0]
                        # print("d : ", abs(x - other_x))
                        if abs(x - other_x) <= self.config["close_vehicle_threshold"]:
                            cost += self.config["close_vehicle_cost"]
                            if self.config["vis"]:
                                print("0 triggered")
                            break
                    elif other_l_index[2] == l_index:
                        x = self.road.network.get_lane(other_l_index).local_coordinates(self.road.vehicles[i].position)[0]
                        other_x = self.road.network.get_lane(other_l_index).local_coordinates(self.road.vehicles[k].position)[0]
                        if x > other_x and (x - other_x <= self.config["close_vehicle_threshold"]):
                            cost += self.config["close_vehicle_cost"]
                            if self.config["vis"]:
                                print("0 triggered")
                            break
                    else:
                        continue

            elif action[j] == 2:
                # lane right
                
                if l_index == self.config["lanes_count"]-1:
                    # change lane is forbidden
                    cost += self.config["invalid_action_cost"]
                    continue
                for k in range(len(self.road.vehicles)):
                    if k == i:
                        continue
                    other_l_index = self.road.vehicles[k].lane_index
                    if other_l_index[2] > l_index and self.road.network.get_lane(other_l_index).is_reachable_from(self.road.vehicles[i].position):
                        x = self.road.network.get_lane(other_l_index).local_coordinates(self.road.vehicles[i].position)[0]
                        other_x = self.road.network.get_lane(other_l_index).local_coordinates(self.road.vehicles[k].position)[0]
                        if abs(x - other_x) <= self.config["close_vehicle_threshold"]:
                            cost += self.config["close_vehicle_cost"]
                            if self.config["vis"]:
                                print("2 triggered")
                            break
                    elif other_l_index[2] == l_index:
                        x = self.road.network.get_lane(other_l_index).local_coordinates(self.road.vehicles[i].position)[0]
                        other_x = self.road.network.get_lane(other_l_index).local_coordinates(self.road.vehicles[k].position)[0]
                        if x > other_x and (x - other_x) <= self.config["close_vehicle_threshold"]:
                            cost += self.config["close_vehicle_cost"]
                            if self.config["vis"]:
                                print("2 triggered")
                            break
                    else:
                        continue
            elif action[j] == 3:
                # faster 
                
                for k in range(len(self.road.vehicles)):
                    if k == i:
                        continue
                    other_l_index = self.road.vehicles[k].lane_index
                    if other_l_index[2] == l_index:
                        x = self.road.network.get_lane(other_l_index).local_coordinates(self.road.vehicles[i].position)[0]
                        other_x = self.road.network.get_lane(other_l_index).local_coordinates(self.road.vehicles[k].position)[0]
                        if other_x > x:
                            if (other_x - x) <= self.config["close_vehicle_threshold"]:
                                cost += self.config["close_vehicle_cost"]
                                if self.config["vis"]:
                                    print("3 triggered")
                                break
                        else:
                            continue
            elif action[j] == 4:
                # slower
                
                for k in range(len(self.road.vehicles)):
                    if k == i:
                        continue
                    other_l_index = self.road.vehicles[k].lane_index
                    if other_l_index[2] == l_index:
                        x = self.road.network.get_lane(other_l_index).local_coordinates(self.road.vehicles[i].position)[0]
                        other_x = self.road.network.get_lane(other_l_index).local_coordinates(self.road.vehicles[k].position)[0]
                        if other_x < x:
                            # TODO: incorporate speed in close vehicle threshold
                            if (x - other_x) <= self.config["close_vehicle_threshold"]:
                                cost += self.config["close_vehicle_cost"]
                                if self.config["vis"]:
                                    print("4 triggered")
                                break
                        else:
                            continue
        if self.config["vis"]:
            print("cost: ", cost)
        return cost
    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            # Forward action to the vehicle
            if self.steps % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                if self.victim_action:
                    self.victim_action_type.act(self.victim_action)
                else:
                    victim_obs = self.victim_observation_type.observe()
                    self.victim_action = self.victim_agent.select_action(victim_obs)
                    self.victim_action_type.act(self.victim_action)
            
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.steps % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)
            

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                if self.config["vis"]:
                    time.sleep(0.5)
                self._automatic_rendering()

        self.enable_auto_render = False

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.viewer is None:
            self.viewer = EnvViewer(self)
            self.viewer.observer_vehicle = self.victim

        self.enable_auto_render = True

        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if self.render_mode == 'rgb_array':
            image = self.viewer.get_image()
            return image

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        # reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        # if self.config["normalize_reward"]:
        #     reward = utils.lmap(reward,
        #                         [self.config["collision_reward"],
        #                          self.config["high_speed_reward"] + self.config["right_lane_reward"]],
        #                         [0, 1])
        # reward *= rewards['on_road_reward']
        # print(rewards)

        self.r_sum += rewards
        return rewards

    def _rewards(self, action: Action) -> Dict[Text, float]:
        # self.vehicle originally is controlled vehicle [0]
        reward = 0.0
        for vehicle in self.controlled_vehicles:
            if vehicle.destroyed:
                continue
            if vehicle.crashed:
                vehicle.destroyed = True
                if vehicle.collide_with != self.victim:
                    print("attacker crashed with each other")
                    reward += self.config["attacker_collide_each_other_reward"]
        reward += self.config["time_penalty"] * self.time
        if self.victim.crashed:
            print("################################################")
            print("victim crashed")
            print("################################################")
            reward += self.config["vicitm_collision_reward"]
        
        return reward
            

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

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        # dones = []
        # if self.time >= self.config["duration"] or self.victim.crashed:
        #     return [True, True, True, True]
        
        # for vehicle in self.controlled_vehicles:
        #     if vehicle.crashed:
        #         dones.append(True)
        #     else:
        #         dones.append(False)
        # return dones
        if (self.victim.crashed or
                self.config["offroad_terminal"] and not self.victim.on_road or
                self.time >= self.config["duration"]):
            return True
        else:
            return False
        # return (self.victim.crashed or
        #         self.config["offroad_terminal"] and not self.victim.on_road or
        #         self.time >= self.config["duration"])

    def _is_truncated(self) -> bool:
        return False
    
    def _info(self, obs: Observation, action: Optional[Action] = None, done: List[bool]=None) -> dict:
        if done is not None and done:
            info = {'episode':{'r': self.r_sum,
                               'l': self.steps,
                               'c': self.c_sum}}
            return info
        else:
            return {}

    # def _get_victim_action(self, victim_state):
    #     # this method should return victim's action given victim's state
    #     return self.victim_predict_fn(victim_state)



class HighwayEnvCentralizedPenaltyFast(HighwayEnvCentralizedPenalty):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 10,  # [s]
            "ego_spacing": 0.5,
        })
        return cfg

