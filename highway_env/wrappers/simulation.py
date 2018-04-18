from __future__ import division, print_function

from highway_env.envs.graphics import EnvViewer
from highway_env.wrappers.graphics import SimulationViewer


class Simulation(object):
    """
        A simulation is the coupling of an environment and an agent, running in closed loop.

        It has its own viewer that allows to display both the representation of the environment and the agent reasoning.
    """
    TRAJECTORY_TIMESTEP = 0.35

    def __init__(self, env, agent, highway_env=None, render_agent=True, env_seed=None, episodes=1):
        """

        :param env: The environment to be solved, possibly wrapping an AbstractEnv environment
        :param agent: The agent solving the environment
        :param AbstractEnv highway_env: if different from env, the wrapped AbstractEnv
        :param render_agent: Whether the agent should be rendered in the Viewer
        :param env_seed: the seed used for the environment randomness source
        """
        self.env = env
        self.agent = agent
        self.highway_env = highway_env if highway_env else env
        self.render_agent = render_agent
        self.env_seed = env_seed
        self.episodes = episodes
        self.planned_trajectory = []
        self.observation = None
        self.steps = 0

        # If agent rendering is requested, create or replace the environment viewer by a simulation viewer
        if self.render_agent:
            self.highway_env.viewer = SimulationViewer(self)

    def run(self):
        self.run_episodes()
        self.close()

    def run_episodes(self):
        for _ in range(self.episodes):
            # Run episode
            terminal = False
            self.env.seed(self.env_seed)
            self.observation = self.env.reset()
            while not terminal:
                # Step until a terminal step is reached
                terminal = self.step()
                self.render()

                # Catch interruptions
                if self.highway_env.done:
                    return

    def step(self):
        """
            Plan a sequence of actions according to the agent policy, and step the environment accordingly.
        """
        # Query agent for actions sequence
        actions = self.agent.plan(self.observation)
        self.planned_trajectory = self.highway_env.vehicle.predict_trajectory(
            [self.highway_env.ACTIONS[a] for a in actions],
            1 / self.highway_env.POLICY_FREQUENCY,
            self.TRAJECTORY_TIMESTEP,
            1 / self.highway_env.SIMULATION_FREQUENCY)
        if not actions:
            raise Exception("The agent did not plan any action")

        # Step the environment
        self.observation, reward, terminal, info = self.env.step(actions[0])
        return terminal

    def render(self, mode='human'):
        """
            Render the environment.
        :param mode: the rendering mode
        """
        self.env.render(mode)

    def close(self):
        """
            Close the simulation.
        """
        self.env.close()
