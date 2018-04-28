from __future__ import division, print_function
import multiprocessing
import gym

import highway_env
from highway_env.agent.mcts import MCTSAgent
from highway_env.agent.ttc_vi import TTCVIAgent
from highway_env.wrappers.simulation import Simulation
from highway_env.wrappers.monitor import MonitorV2


def test():
    env = gym.make('highway-merge-v0')
    monitor = MonitorV2(env, 'out', force=True)
    agent = MCTSAgent(prior_policy=MCTSAgent.fast_policy,
                      rollout_policy=MCTSAgent.idle_policy,
                      iterations=50,
                      assume_vehicle_type=None)
    # agent = TTCVIAgent()
    sim = Simulation(monitor, agent, highway_env=env, episodes=10)
    sim.run()


if __name__ == '__main__':
    for i in range(4):
        p = multiprocessing.Process(target=test)
        p.start()
