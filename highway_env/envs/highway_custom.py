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

Observation = np.ndarray


class HighwayEnvCustom(AbstractEnv):
    victim = None
    victim_action = None
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
            self.victim_action_type.controlled_vehicle = self.victim
        self.observation_space = self.observation_type.space()
        # self.action_space = spaces.MultiDiscrete([len(self.attacker_action_type.actions) for _ in range(self.config["attacker_num"])])
        
        self.action_type = action_factory(self, self.config["action"])
        self.action_space = self.action_type.space()

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "AttackerKinematics",
                    "see_behind": True
                }
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
            "offroad_terminal": False
        })
        return config
    
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        victim_index = (self.config["controlled_vehicles"]+1)//2
        for i in range(self.config["controlled_vehicles"]+1):
            if i == victim_index:
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

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        victim_obs = self.victim_observation_type.observe()
        self.victim_action = self.victim_agent.select_action(victim_obs)
        print("victim action: {}".format(self.victim_action))
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            # Forward action to the vehicle
            if self.victim_action:
                self.victim_action_type.act(self.victim_action)
            else:
                victim_obs = self.victim_observation_type.observe()
                self.victim_action_type.act(self.victim_agent.select_action(victim_obs))
            
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
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road or
                self.time >= self.config["duration"])

    def _is_truncated(self) -> bool:
        return False

    def _get_victim_action(self, victim_state):
        # this method should return victim's action given victim's state
        return self.victim_predict_fn(victim_state)



class HighwayEnvCustomFast(HighwayEnvCustom):
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
            "duration": 60,  # [s]
            "ego_spacing": 0.5,
        })
        return cfg
