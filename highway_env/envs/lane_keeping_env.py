from __future__ import division, print_function, absolute_import

import copy

import numpy as np
from gym import spaces
from gym.envs.registration import register

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.dynamics import BicycleVehicle


class LaneKeepingEnv(AbstractEnv):
    """
        A lane keeping control task.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.lane = None
        self.trajectory = []
        self.interval_trajectory = []
        self.lpv = None

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
            "state_noise": 0.05,
            "derivative_noise": 0.05,
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
        self.store_data()
        if self.lpv:
            self.lpv.set_control(control=action.squeeze(-1),
                                 state=self.vehicle.state[[1, 2, 4, 5]])
            self.lpv.step(1 / self.config["simulation_frequency"])

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
        _, lat = self.lane.local_coordinates(self.vehicle.position)
        return 1 - (lat/self.lane.width)**2

    def _is_terminal(self):
        return False  # not self.lane.on_lane(self.vehicle.position)

    def reset(self):
        self._make_road()
        self._make_vehicles()
        return super().reset()

    def _make_road(self):
        net = RoadNetwork()

        self.lane = SineLane([0, 0], [500, 0], amplitude=5, pulsation=2*np.pi / 100, phase=0,
                        width=10, line_types=[LineType.STRIPED, LineType.STRIPED])
        # self.lane = StraightLane([0, 0], [500, 0], line_types=[LineType.STRIPED, LineType.STRIPED], width=10)
        net.add_lane("a", "b", self.lane)
        other_lane = StraightLane([50, 50], [120, 15], line_types=[LineType.STRIPED, LineType.STRIPED], width=10)
        # other_lane = StraightLane([0, 5], [500, 5], line_types=[LineType.STRIPED, LineType.STRIPED], width=10)
        net.add_lane("c", "d", other_lane)
        net.add_lane("d", "a", StraightLane([120, 15], [120+20, 15+20*(15-50)/(120-50)], line_types=[LineType.NONE, LineType.STRIPED], width=10))
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self):
        road = self.road
        ego_vehicle = BicycleVehicle(road, road.network.get_lane(("c", "d", 0)).position(80, 0),
                                     heading=road.network.get_lane(("c", "d", 0)).heading_at(30),
                                     velocity=8.3)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

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
        longi, lat = self.lane.local_coordinates(self.vehicle.position)
        psi_l = self.lane.heading_at(longi)
        state = self.vehicle.state[[1, 2, 4, 5]]
        return np.array([[state[0, 0] - lat], [psi_l], [0], [0]])

    def store_data(self):
        if self.lpv:
            state = self.vehicle.state.copy()
            interval = []
            for x_t in self.lpv.change_coordinates(self.lpv.x_i_t, back=True, interval=True):
                # lateral state to full state
                np.put(state, [1, 2, 4, 5], x_t)
                # full state to absolute coordinates
                interval.append(state.squeeze(-1).copy())
            self.interval_trajectory.append(interval)
        self.trajectory.append(copy.deepcopy(self.vehicle.state))


register(
    id='lane-keeping-v0',
    entry_point='highway_env.envs:LaneKeepingEnv',
    max_episode_steps=200
)
