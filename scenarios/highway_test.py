from __future__ import division, print_function

import gym
import highway_env

from highway_env.agent.mcts import MCTSAgent
from highway_env.simulation.simulation import Simulation


def test():
    env = gym.make('highway-v0')
    agent = MCTSAgent(env)
    sim = Simulation(env, agent)

    sim.render()
    while not sim.done:
        sim.step()
    sim.close()


if __name__ == '__main__':
    for _ in range(1):
        test()
