from __future__ import division, print_function

import gym
import highway_env

from highway_env.agent.mcts import MCTSAgent
from highway_env.agent.ttc_vi import TTCVIAgent
from highway_env.simulation.simulation import Simulation


def test():
    env = gym.make('highway-v0')
    env.vehicle.position[0] -= 70
    env.vehicle.position[1] = 3*4
    env.vehicle.target_lane_index = 3
    agent = MCTSAgent(env, temperature=200, iterations=100)
    sim = Simulation(env, agent)

    sim.step()
    sim.render()
    while not sim.done:
        sim.step()
    sim.close()


if __name__ == '__main__':
    for _ in range(10):
        test()
