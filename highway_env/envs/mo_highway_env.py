import numpy as np
from typing import Tuple
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray

class MOHighwayEnv(AbstractEnv):
    """
    Multi-objective version of HighwayEnv
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
            "lanes_count": 4,
            "vehicles_count": 50,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "reward_speed_range": [20, 30],
            "offroad_terminal": False,
            "cur_reward": 0
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if np.random.randint(0,10) == 0:
            print("Adding new vehicle!") 
            self._add_vehicle()
        return AbstractEnv.step(self, action)

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        # Create controlled vehicle
        self.controlled_vehicles = []
        vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
        vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.append(vehicle)

        # Create other vehicles
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)
    
    def _add_vehicle(self) -> None:
        """Add vehicles to the left of the screen if vehicle speed is too slow"""
        _from = self.road.np_random.choice(list(self.road.network.graph.keys()))
        _to = self.road.np_random.choice(list(self.road.network.graph[_from].keys()))

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        lanes = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane_id = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        lane = self.road.network.get_lane((_from, _to, lane_id))
        default_spacing = 12+1.0*self.vehicle.speed        
        offset = default_spacing * np.exp(-5 / 40 * len(lanes))
        x0 = lane.local_coordinates(self.vehicle.position)
        x0 -= offset * self.road.np_random.uniform(0.9, 1.1)

        spawn_lane_id = np.random.randint(0,max(len(lanes) - 1, 1))
        spawn_lane = self.road.network.get_lane((_from, _to, spawn_lane_id))
        spawn_position = spawn_lane.position(x0,0)
        spawn_speed = self.vehicle.speed * 1.25

        vehicle = other_vehicles_type(road=self.road, position=spawn_position, speed=spawn_speed)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        :param action: the last action performed
        :return: the corresponding reward
        """
        # SPEED OBJECTIVE
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        speed_reward = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

        # RIGHT LANE OBJECTIVE
        lanes = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        right_reward = lane / max(len(lanes) - 1, 1)

        # DON'T CRASH OBJECTIVE
        safe_reward = 0 if self.vehicle.crashed \
            else 1

        reward_vector = [speed_reward, right_reward, safe_reward]

        reward = 0 if not self.vehicle.on_road else reward_vector[self.config["cur_reward"]]
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.time >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

register(
    id='mo-highway-v0',
    entry_point='highway_env.envs:MOHighwayEnv',
)