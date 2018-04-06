import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from highway_env.mdp.road_mdp import RoadMDP
from highway_env.road.road import Road
from highway_env.simulation.graphics import SimulationWindow
from highway_env.simulation.simulation import Simulation
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.control import MDPVehicle


class HighwayEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        road = Road.create_random_road(lanes_count=4, lane_width=4.0, vehicles_count=20, vehicles_type=IDMVehicle)
        self.sim = Simulation(road, ego_vehicle_type=MDPVehicle, agent_type=None)
        self.viewer = None
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    def step(self, action):
        self.sim.act()
        self.sim.vehicle.act(RoadMDP.ACTIONS[action])
        self.sim.step()
        self.sim.ending_criterion()

        ob = 1
        reward = 1
        done = self.sim.done

        return ob, reward, done, {}

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            return None  # Unsupported for now
        elif mode == 'human':
            self._get_viewer().handle_events()
            self._get_viewer().display()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = SimulationWindow(self.sim, agent_displayed=False, record_video=False)
        return self.viewer

    def close(self):
        if self.viewer is not None:
            self.viewer.quit()
        self.viewer = None
