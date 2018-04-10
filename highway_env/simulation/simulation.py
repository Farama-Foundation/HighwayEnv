from __future__ import division, print_function
import numpy as np

from highway_env.simulation.graphics import SimulationViewer


class Simulation(object):
    MAXIMUM_SIMULATION_TIME = 3 * 60
    TRAJECTORY_TIMESTEP = 0.35

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.planned_trajectory = []
        self.done = False

    def step(self):
        actions = self.agent.plan(self.env.simplified())
        self.planned_trajectory = self.env.vehicle.predict_trajectory([self.env.ACTIONS[a] for a in actions],
                                                                      1 / self.env.POLICY_FREQUENCY,
                                                                      self.TRAJECTORY_TIMESTEP,
                                                                      1 / self.env.SIMULATION_FREQUENCY)
        if actions:
            _, reward, terminal, _ = self.env.step(actions[0])
            self.done = terminal or self.env.done

    def render(self, mode='human'):
        if mode == 'rgb_array':
            raise NotImplementedError()
        elif mode == 'human':
            self._get_viewer().handle_events()
            self._get_viewer().display()

    def _get_viewer(self):
        if self.env.viewer is None:
            self.env.viewer = SimulationViewer(self, record_video=False)
        return self.env.viewer

    def close(self):
        self.env.close()
