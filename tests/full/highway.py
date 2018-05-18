from __future__ import division, print_function
import multiprocessing
import gym

from highway_env.vehicle.behavior import IDMVehicle
from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.wrappers.monitor import MonitorV2
from highway_env.wrappers.simulation import Simulation


def test():
    IDMVehicle.POLITENESS = 0
    env = gym.make('highway-v0')
    monitor = MonitorV2(env, 'out', force=True)
    agent = MCTSAgent(temperature=30, iterations=100)
    # agent = TTCVIAgent()
    sim = Simulation(monitor, agent, highway_env=env, episodes=25, sim_seed=None)
    sim.run()


if __name__ == '__main__':
    for i in range(4):
        p = multiprocessing.Process(target=test)
        p.start()
