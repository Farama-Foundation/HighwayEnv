from math import floor, ceil
import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import MOAbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

class MOHighwayEnv(MOAbstractEnv):
    """
    Multi-objective version of HighwayEnv
    """
    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self._add_reward("speed", self._speed_reward)
        self._add_reward("right", self._right_reward)
        self._add_reward("nocrash", self._nocrash_reward)
        # self._add_reward("acceleration", self._acc_reward)
        self._add_reward("distance", self._dist_reward)

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
        for _ in range(ceil(self.config["vehicles_count"] * 0.5)):
            self._add_vehicle_front()
        for _ in range(floor(self.config["vehicles_count"] * 0.5)):
            self._add_vehicle_behind()
    
    def _add_vehicle_front(self) -> None:
        """Add vehicle in front of leading car"""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"], front=True)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

    def _add_vehicle_behind(self) -> None:
        """Add vehicle behind last car"""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"], front=False)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

    def _speed_reward(self, action: Action) -> float:
        # SPEED
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        return utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

    def _right_reward(self, action: Action) -> float:
        # RIGHT LANE
        lanes = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        return lane / max(len(lanes) - 1, 1)

    def _nocrash_reward(self, action: Action) -> float:
        # DON'T CRASH
        return 0 if self.vehicle.crashed else 1

    def _acc_reward(self, action: Action) -> float:
        # MINIMIZE ACCELERATION (fuel efficiency?)
        acc_max = self.vehicle.ACC_MAX
        front_vehicle, rear_vehicle = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.target_lane_index)
        acc = self.vehicle.acceleration(ego_vehicle=self.vehicle, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
        norm_accel = utils.lmap(acc, [0,acc_max], [0,1])
        return 1-norm_accel

    def _dist_reward(self, action: Action) -> float:
        # MAXIMIZE MIN DISTANCE
        return np.min(
                [   
                    np.linalg.norm(v.position - self.vehicle.position) 
                    for v in self.road.vehicles 
                    if v is not self.vehicle
                ]
            )

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.time >= self.config["duration"] or \
            not self.vehicle.on_road

register(
    id='mo-highway-v0',
    entry_point='highway_env.envs:MOHighwayEnv',
)