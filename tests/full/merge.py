from __future__ import division, print_function
import multiprocessing
import gym

from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.wrappers.monitor import MonitorV2
from highway_env.wrappers.simulation import Simulation


def test():
    env = gym.make('highway-merge-v0')
    monitor = MonitorV2(env, 'out')
    agent = MCTSAgent(prior_policy=MCTSAgent.fast_policy,
                      rollout_policy=MCTSAgent.idle_policy,
                      iterations=50,
                      assume_vehicle_type=None)
    sim = Simulation(monitor, agent, highway_env=env, episodes=10)
    sim.run()


if __name__ == '__main__':
    for i in range(4):
        p = multiprocessing.Process(target=test)
        p.start()
