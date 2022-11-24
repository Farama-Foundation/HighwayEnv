from __future__ import division, print_function, absolute_import

import copy
from typing import Tuple

import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.dynamics import BicycleVehicle


class LaneKeepingEnv(AbstractEnv):

    """A lane keeping control task."""

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.lane = None
        self.lanes = []
        self.trajectory = []
        self.interval_trajectory = []
        self.lpv = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "AttributesObservation",
                "attributes": ["state", "derivative", "reference_state"]
            },
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi / 3, np.pi / 3],
                "longitudinal": False,
                "lateral": True,
                "dynamical": True
            },
            "simulation_frequency": 10,
            "policy_frequency": 10,
            "state_noise": 0.05,
            "derivative_noise": 0.05,
            "screen_width": 600,
            "screen_height": 250,
            "scaling": 7,
            "centering_position": [0.4, 0.5]
        })
        return config

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.lanes and not self.lane.on_lane(self.vehicle.position):
            self.lane = self.lanes.pop(0)
        self.store_data()
        if self.lpv:
            self.lpv.set_control(control=action.squeeze(-1),
                                 state=self.vehicle.state[[1, 2, 4, 5]])
            self.lpv.step(1 / self.config["simulation_frequency"])

        self.action_type.act(action)
        obs = self.observation_type.observe()
        self._simulate()

        info = {}
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        return obs, reward, terminated, truncated, info

    def _reward(self, action: np.ndarray) -> float:
        _, lat = self.lane.local_coordinates(self.vehicle.position)
        return 1 - (lat/self.lane.width)**2

    def _is_terminated(self) -> bool:
        return False

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()
        lane = SineLane([0, 0], [500, 0], amplitude=5, pulsation=2*np.pi / 100, phase=0,
                        width=10, line_types=[LineType.STRIPED, LineType.STRIPED])
        net.add_lane("a", "b", lane)
        other_lane = StraightLane([50, 50], [115, 15],
                                  line_types=(LineType.STRIPED, LineType.STRIPED), width=10)
        net.add_lane("c", "d", other_lane)
        self.lanes = [other_lane, lane]
        self.lane = self.lanes.pop(0)
        net.add_lane("d", "a", StraightLane([115, 15], [115+20, 15+20*(15-50)/(115-50)],
                                            line_types=(LineType.NONE, LineType.STRIPED), width=10))
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("c", "d", 0)).position(50, -4),
            heading=road.network.get_lane(("c", "d", 0)).heading_at(0),
            speed=8.3)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    @property
    def dynamics(self) -> BicycleVehicle:
        return self.vehicle

    @property
    def state(self) -> np.ndarray:
        if not self.vehicle:
            return np.zeros((4, 1))
        return self.vehicle.state[[1, 2, 4, 5]] + \
               self.np_random.uniform(low=-self.config["state_noise"],
                                      high=self.config["state_noise"],
                                      size=self.vehicle.state[[0, 2, 4, 5]].shape)

    @property
    def derivative(self) -> np.ndarray:
        if not self.vehicle:
            return np.zeros((4, 1))
        return self.vehicle.derivative[[1, 2, 4, 5]] + \
               self.np_random.uniform(low=-self.config["derivative_noise"],
                                      high=self.config["derivative_noise"],
                                      size=self.vehicle.derivative[[0, 2, 4, 5]].shape)

    @property
    def reference_state(self) -> np.ndarray:
        if not self.vehicle or not self.lane:
            return np.zeros((4, 1))
        longi, lat = self.lane.local_coordinates(self.vehicle.position)
        psi_l = self.lane.heading_at(longi)
        state = self.vehicle.state[[1, 2, 4, 5]]
        return np.array([[state[0, 0] - lat], [psi_l], [0], [0]])

    def store_data(self) -> None:
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
