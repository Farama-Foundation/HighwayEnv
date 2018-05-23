from __future__ import division, print_function
import multiprocessing
import gym

from highway_env.vehicle.behavior import IDMVehicle
from rl_agents.agents.dqn.dqn_pytorch import DQNPytorchAgent
from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.trainer.simulation import Simulation


def dqn_pytorch(environment):
    config = {
        "layers": [256, 256],
        "memory_capacity": 50000,
        "batch_size": 100,
        "gamma": 0.9,
        "epsilon": [1.0, 0.01],
        "epsilon_tau": 50000*2,
        "target_update": 1
    }
    return DQNPytorchAgent(environment, config)


def configure_environment(environment, level):
    if level == "EASY":
        environment.LANES_COUNT = 2
        environment.INITIAL_SPACING = 1
        environment.VEHICLES_COUNT = 5
        environment.DURATION = 20


def main():
    gym.logger.set_level(gym.logger.INFO)
    IDMVehicle.POLITENESS = 0
    env = gym.make('highway-v0')
    # configure_environment(env, "EASY")

    agent = MCTSAgent(env, temperature=50, iterations=50, max_depth=7)
    # agent = TTCVIAgent()
    # agent = dqn_pytorch(env)
    sim = Simulation(env, agent, num_episodes=5000*2, sim_seed=None, recover=True)
    sim.test()


if __name__ == '__main__':
    main()
    # for i in range(4):
    #     p = multiprocessing.Process(target=tests)
    #     p.start()
