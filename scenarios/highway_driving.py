import gym
import highway_env

from highway_env.agent.mcts import MCTSAgent
from highway_env.simulation.simulation import Simulation


def test():
    env = gym.make('highway-v0')
    agent = MCTSAgent(env)
    sim = Simulation(env, agent)

    sim.render()
    for _ in range(5):
        sim.step()


if __name__ == '__main__':
    for _ in range(1):
        test()
