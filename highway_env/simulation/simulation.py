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

    def step(self):
        actions = self.agent.plan(self.env.simplified())
        self.planned_trajectory = self.env.vehicle.predict_trajectory(actions,
                                                                      1 / self.env.POLICY_FREQUENCY,
                                                                      self.TRAJECTORY_TIMESTEP,
                                                                      1 / self.env.SIMULATION_FREQUENCY)
        if actions:
            self.env.step(actions[0])

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


    # def ending_criterion(self):
    #     if self.t/self.SIMULATION_FREQUENCY > self.MAXIMUM_SIMULATION_TIME \
    #             or (self.vehicle.crashed and self.vehicle.velocity < 1) \
    #             or ((len(self.road.vehicles) > 1)
    #                 and (self.vehicle.position[0] > 50 + np.max(
    #                     [o.position[0] for o in self.road.vehicles if o is not self.vehicle]))):
    #         self.done = True
