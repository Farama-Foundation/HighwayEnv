from __future__ import division, print_function, absolute_import
import numpy as np
from gym import spaces
from gym.envs.registration import register

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.dynamics import BicycleVehicle


class LaneKeepingEnv(AbstractEnv):
    """
        A lane keeping control task.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "AttributesObservation",
                "attributes": ["state", "derivative", "reference_state"]
            },
            "simulation_frequency": 10,
            "policy_frequency": 10,
            "steering_range": np.pi / 3,
            "state_noise": 0.1,
            "derivative_noise": 0.1,
            "screen_width": 600,
            "screen_height": 300,
            "scaling": 7,
            "centering_position": [0.4, 0.5]
        })
        return config

    def define_spaces(self):
        super().define_spaces()
        self.action_space = spaces.Box(-self.config["steering_range"], self.config["steering_range"], shape=(1,), dtype=np.float32)

    def step(self, action):
        self.vehicle.act({
            "acceleration": 0,
            "steering": action[0]
        })
        obs = self.observation.observe()
        self._simulate()

        info = {}
        reward = self._reward(action)
        terminal = self._is_terminal()
        return obs, reward, terminal, info

    def _reward(self, action):
        _, lat = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return 1 - (lat/self.vehicle.lane.width)**2

    def _is_terminal(self):
        return False  # not self.vehicle.lane.on_lane(self.vehicle.position)

    def reset(self):
        self._make_road()
        self._make_vehicles()
        return super().reset()

    def _make_road(self):
        net = RoadNetwork()

        lane = SineLane([0, 0], [500, 0], amplitude=5, pulsation=2*np.pi / 50, phase=0,
                        width=10, line_types=[LineType.STRIPED, LineType.STRIPED])
        net.add_lane("a", "b", lane)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self):
        road = self.road
        ego_vehicle = BicycleVehicle(road, road.network.get_lane(("a", "b", 0)).position(30, 10),
                                     velocity=8.3)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        print(ego_vehicle.full_lateral_lpv_structure())

    @property
    def dynamics(self):
        return self.vehicle

    @property
    def state(self):
        return self.vehicle.state[[1, 2, 4, 5]] + \
               self.np_random.uniform(low=-self.config["state_noise"],
                                      high=self.config["state_noise"],
                                      size=self.vehicle.state[[0, 2, 4, 5]].shape)

    @property
    def derivative(self):
        return self.vehicle.derivative[[1, 2, 4, 5]] + \
               self.np_random.uniform(low=-self.config["derivative_noise"],
                                      high=self.config["derivative_noise"],
                                      size=self.vehicle.derivative[[0, 2, 4, 5]].shape)

    @property
    def reference_state(self):
        longi, lat = self.vehicle.lane.local_coordinates(self.vehicle.position)
        psi_l = self.vehicle.lane.heading_at(longi)
        state = self.vehicle.state[[1, 2, 4, 5]]
        return np.array([[lat - state[0, 0]], [psi_l], [0], [0]])


register(
    id='lane-keeping-v0',
    entry_point='highway_env.envs:LaneKeepingEnv',
    max_episode_steps=200
)
