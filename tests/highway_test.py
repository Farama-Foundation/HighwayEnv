from __future__ import division, print_function

import gym
from highway_env.wrappers.monitor import MonitorV2

import highway_env

from highway_env.agent.mcts import MCTSAgent
from highway_env.agent.ttc_vi import TTCVIAgent
from highway_env.wrappers.simulation import Simulation


def test():
    env = gym.make('highway-v0')
    monitor = MonitorV2(env, 'out', force=True)
    # agent = MCTSAgent(env, temperature=200, iterations=100)
    observation = monitor.reset()
    agent = TTCVIAgent(observation)
    sim = Simulation(monitor, agent, observation, highway_env=env, episodes=3)

    while not sim.done:
        sim.step()
        sim.render()
    sim.close()


if __name__ == '__main__':
    test()
