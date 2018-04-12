from __future__ import division, print_function

from highway_env.envs.graphics import EnvViewer
from highway_env.wrappers.graphics import SimulationViewer


class Simulation(object):
    """
        A simulation is the coupling of an environment and an agent, running in closed loop.

        It has its own viewer that allows to display both the representation of the environment and the agent reasoning.
    """
    MAXIMUM_SIMULATION_TIME = 3 * 60
    TRAJECTORY_TIMESTEP = 0.35

    def __init__(self, env, agent, initial_observation, highway_env=None, render_agent=True, episodes=1):
        """

        :param env: The environment to be solved, possibly wrapping an AbstractEnv environment
        :param agent: The agent solving the environment
        :param AbstractEnv highway_env: if different from env, the wrapped AbstractEnv
        :param render_agent: Whether the agent should be rendered in the Viewer
        """
        self.env = env
        self.agent = agent
        self.observation = initial_observation
        self.highway_env = highway_env if highway_env else env
        self.render_agent = render_agent
        self.episodes = episodes
        self.planned_trajectory = []
        self.done = False
        self.steps = 0

        if self.render_agent:
            self.highway_env.viewer = SimulationViewer(self, record_video=False)

    def step(self):
        """
            Plan a sequence of actions according to the agent policy, and step the environment accordingly.
        """
        actions = self.agent.plan(self.observation)
        self.planned_trajectory = self.highway_env.vehicle.predict_trajectory(
            [self.highway_env.ACTIONS[a] for a in actions],
            1 / self.highway_env.POLICY_FREQUENCY,
            self.TRAJECTORY_TIMESTEP,
            1 / self.highway_env.SIMULATION_FREQUENCY)

        if actions:
            self.observation, reward, terminal, _ = self.env.step(actions[0])
            if terminal:
                self.next_episode()

        self.steps += 1
        if self.steps > self.MAXIMUM_SIMULATION_TIME:
            self.next_episode()

        self.done = self.done or self.highway_env.done

    def next_episode(self):
        self.episodes -= 1
        if self.episodes > 0:
            self.env.reset()
        else:
            self.done = True

    def render(self, mode='human'):
        """
            Render the environment.

            Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """

        if self.highway_env.viewer is None:
            if self.render_agent:
                self.highway_env.viewer = SimulationViewer(self, record_video=False)
            else:
                self.highway_env.viewer = EnvViewer(self.env, record_video=False)
        self.env.render(mode)

    def close(self):
        """
            Close the simulation.
        """
        self.env.close()
