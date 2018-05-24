from __future__ import division, print_function
import multiprocessing
import gym

from highway_env.vehicle.behavior import IDMVehicle
from rl_agents.agents.dqn.dqn_pytorch import DQNPytorchAgent
from rl_agents.agents.tree_search.mcts import MCTSAgent, MCTSWithPriorPolicyAgent
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


def mcts(environment):
    fast_policy = lambda state: MCTSAgent.preference_policy(state, environment.ACTIONS_INDEXES['FASTER'])
    return MCTSAgent(environment,
                     prior_policy=fast_policy,
                     rollout_policy=MCTSAgent.random_available_policy,
                     iterations=75,
                     temperature=10,
                     max_depth=7)


def mcts_with_prior(environment):
    config = {
        "layers": [256, 256],
        "memory_capacity": 50000,
        "batch_size": 100,
        "gamma": 0.9,
        "epsilon": [0.3, 0.3],
        "epsilon_tau": 50000*2,
        "target_update": 1
    }
    prior_agent = DQNPytorchAgent(environment, config)
    prior_agent.load('out/HighwayEnv/DQNPytorchAgent/saved_models/dqn_256_256_50000.tar')
    return MCTSWithPriorPolicyAgent(environment,
                                    prior_agent=prior_agent,
                                    iterations=25,
                                    temperature=10,
                                    max_depth=7)


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
    configure_environment(env, "EASY")

    # agent = TTCVIAgent()
    # agent = dqn_pytorch(env)
    # agent = mcts(env)
    agent = mcts_with_prior(env)
    sim = Simulation(env, agent, num_episodes=80, sim_seed=None)
    sim.test()


if __name__ == '__main__':
    for i in range(4):
        p = multiprocessing.Process(target=main)
        p.start()
    # main()
