from __future__ import division, print_function
import multiprocessing
import gym

from highway_env.vehicle.behavior import IDMVehicle
from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.trainer.simulation import Simulation


def test():
    gym.logger.set_level(gym.logger.INFO)
    IDMVehicle.POLITENESS = 0
    env = gym.make('highway-v0')
    agent = MCTSAgent(env, temperature=30, iterations=50)
    # agent = TTCVIAgent()
    sim = Simulation(env, agent, num_episodes=25, sim_seed=None)
    sim.test()


if __name__ == '__main__':
    for i in range(4):
        p = multiprocessing.Process(target=test)
        p.start()
