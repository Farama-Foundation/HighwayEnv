from __future__ import division, print_function

import gym
import highway_env

from highway_env.agent.mcts import MCTSAgent
from highway_env.simulation.simulation import Simulation


def run():
    env = gym.make('highway-merge-v0')
    agent = MCTSAgent(env,
                      prior_policy=MCTSAgent.fast_policy,
                      rollout_policy=MCTSAgent.idle_policy,
                      iterations=100,
                      assume_vehicle_type=None)
    sim = Simulation(env, agent)

    sim.render()
    while not sim.done:
        sim.step()
    sim.close()


if __name__ == '__main__':
    # np.random.seed(3)
    run()
