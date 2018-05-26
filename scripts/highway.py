from __future__ import division, print_function
import multiprocessing
import gym

from highway_env.vehicle.behavior import IDMVehicle
from rl_agents.agents.dqn.dqn_pytorch import DQNPytorchAgent
from rl_agents.agents.exploration.boltzmann import Boltzmann
from rl_agents.agents.exploration.epsilon_greedy import EpsilonGreedy
from rl_agents.agents.tree_search.mcts import MCTSAgent, MCTSWithPriorPolicyAgent
from rl_agents.configuration import Config
from rl_agents.trainer.benchmark import Benchmark
from rl_agents.trainer.simulation import Simulation


def main():
    env = prepare_environment()
    evaluate_agent(env)
    # benchmark_agents(env)


def prepare_environment():
    gym.logger.set_level(gym.logger.INFO)
    env = gym.make('highway-v0')
    env.set_difficulty_level("EASY")
    return env


def evaluate_agent(env):
    # agent = TTCVIAgent()
    agent = dqn_pytorch(env)
    # agent = mcts(env)
    # agent = mcts_with_prior(env)
    sim = Simulation(env, agent, num_episodes=80, sim_seed=None)
    sim.test()


def benchmark_agents(env):
    # agents = [mcts(env, iterations=20), mcts(env, iterations=60), mcts(env, iterations=100)]
    agents = [mcts_with_prior(env, temperature=0.0),
              mcts_with_prior(env, temperature=0.1),
              mcts_with_prior(env, temperature=0.3),
              mcts_with_prior(env, temperature=0.5),
              mcts_with_prior(env, temperature=0.7),
              mcts_with_prior(env, temperature=1.0)]
    benchmark = Benchmark(env, agents, num_episodes=25)
    benchmark.run()


def dqn_pytorch(environment):
    config = Config(layers=[256, 256],
                    memory_capacity=50000,
                    batch_size=100,
                    gamma=0.9,
                    exploration=Config(method=EpsilonGreedy,
                                       epsilon=[1.0, 0.1],
                                       epsilon_tau=50000),
                    target_update=1)
    return DQNPytorchAgent(environment, config)


def mcts(environment, iterations=75, temperature=10):
    from functools import partial
    fast_policy = partial(MCTSAgent.preference_policy, action_index=environment.ACTIONS_INDEXES['FASTER'])
    return MCTSAgent(environment,
                     prior_policy=fast_policy,
                     rollout_policy=MCTSAgent.random_available_policy,
                     iterations=iterations,
                     temperature=temperature,
                     max_depth=7)


def mcts_with_prior(environment, temperature=0.5):
    config = Config(layers=[256, 256],
                    memory_capacity=50000,
                    batch_size=100,
                    gamma=0.9,
                    exploration=Config(method=Boltzmann,
                                       temperature=temperature),
                    target_update=1)
    prior_agent = DQNPytorchAgent(environment, config)
    prior_agent.load('out/HighwayEnv/DQNPytorchAgent/saved_models/easy_15000.tar')
    return MCTSWithPriorPolicyAgent(environment,
                                    prior_agent=prior_agent,
                                    iterations=25,
                                    temperature=10,
                                    max_depth=7)


if __name__ == '__main__':
    num_processes = 1
    if num_processes > 1:
        for i in range(4):
            p = multiprocessing.Process(target=main)
            p.start()
    else:
        main()
