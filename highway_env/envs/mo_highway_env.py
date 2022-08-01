import numpy as np
from gym.envs.registration import register

from highway_env import HighwayEnv, utils
from highway_env.envs.common.abstract import MOAbstractEnv
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle

class MOHighwayEnv(MOAbstractEnv, HighwayEnv):
    """
    A multi-objective version of HighwayEnv
    """
    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self._add_reward("collision", self._collision_reward)
        self._add_reward("right_lane", self._right_lane_reward)
        self._add_reward("high_speed", self._high_speed_reward)

    def _collision_reward(self, action: Action) -> float:
        return self.vehicle.crashed
    
    def _right_lane_reward(self, action: Action) -> float:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        return lane / max(len(neighbours) - 1, 1)

    def _high_speed_reward(self, action: Action) -> float:
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return np.clip(scaled_speed, 0, 1)

    def _utility(self, rewards:dict) -> float:
        """
        The utility is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param rewards: the vector of different objective rewards
        :return: a scalarized utility
        """
        reward = \
            + self.config["collision_reward"] * rewards["collision"] \
            + self.config["right_lane_reward"] * rewards["right_lane"] \
            + self.config["high_speed_reward"] * rewards["high_speed"]
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

class MOHighwayEnvFast(MOHighwayEnv):
    """
    A multi-objective version of HighwayEnvFast
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


register(
    id='mo-highway-v0',
    entry_point='highway_env.envs:MOHighwayEnv',
)

register(
    id='mo-highway-fast-v0',
    entry_point='highway_env.envs:MOHighwayEnvFast',
)
