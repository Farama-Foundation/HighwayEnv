from __future__ import division, print_function

import gym
import highway_env

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.agent.ttc_vi import TTCVIAgent
from highway_env.agent.mcts import MCTSAgent
from highway_env.wrappers.simulation import Simulation


def highway_ttcvi():
    IDMVehicle.POLITENESS = 0
    env = gym.make('highway-v0')
    env.vehicle.position[0] -= 70
    env.vehicle.position[1] = 3*4
    env.vehicle.target_lane_index = 3
    agent = TTCVIAgent(env)
    sim = Simulation(env, agent, render_agent=False)

    sim.step()
    sim.render()
    while not sim.done:
        sim.step()
    sim.close()


def highway_mcts():
    IDMVehicle.POLITENESS = 0.2
    env = gym.make('highway-v0')
    env.vehicle.position[0] -= 70
    env.vehicle.position[1] = 3*4
    env.vehicle.target_lane_index = 3
    agent = MCTSAgent(env, temperature=300, iterations=100)
    sim = Simulation(env, agent)

    sim.step()
    sim.render()
    while not sim.done:
        sim.step()
    sim.close()


def merge():
    IDMVehicle.POLITENESS = 0
    env = gym.make('highway-merge-v0')
    agent = MCTSAgent(env,
                      prior_policy=MCTSAgent.fast_policy,
                      rollout_policy=MCTSAgent.idle_policy,
                      iterations=75,
                      temperature=200,
                      assume_vehicle_type=None)
    sim = Simulation(env, agent, render_agent=False)

    sim.step()
    sim.render()
    while not sim.done:
        sim.step()
    sim.close()


if __name__ == '__main__':
    for _ in range(10):
        highway_mcts()
    for _ in range(10):
        highway_ttcvi()
    for _ in range(0):
        merge()
