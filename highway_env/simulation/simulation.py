from __future__ import division, print_function

from highway_env.simulation.graphics import SimulationViewer


class Simulation(object):
    """
        A simulation is the coupling of an environment and an agent, running in closed loop.

        It has its own viewer that allows to display both the representation of the environment and the agent reasoning.
    """
    MAXIMUM_SIMULATION_TIME = 3 * 60
    TRAJECTORY_TIMESTEP = 0.35

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.planned_trajectory = []
        self.done = False

    def step(self):
        """
            Plan a sequence of actions according to the agent policy, and step the environment accordingly.
        """
        actions = self.agent.plan(self.env.simplified())
        self.planned_trajectory = self.env.vehicle.predict_trajectory([self.env.ACTIONS[a] for a in actions],
                                                                      1 / self.env.POLICY_FREQUENCY,
                                                                      self.TRAJECTORY_TIMESTEP,
                                                                      1 / self.env.SIMULATION_FREQUENCY)
        if actions:
            _, reward, terminal, _ = self.env.step(actions[0])
            self.done = terminal or self.env.done

    def render(self, mode='human'):
        """
            Render the environment.

            Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        if self.env.viewer is None:
            self.env.viewer = SimulationViewer(self, record_video=True)
        if mode == 'rgb_array':
            raise NotImplementedError()
        elif mode == 'human':
            self.viewer.display()
            self.viewer.handle_events()

    def close(self):
        """
            Close the simulation.
        """
        self.env.close()
