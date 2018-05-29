from __future__ import division, print_function
import multiprocessing
import gym

from rl_agents.agents.dqn.dqn_pytorch import DQNPytorchAgent
from rl_agents.agents.tree_search.mcts import MCTSAgent, MCTSWithPriorPolicyAgent
from rl_agents.trainer.benchmark import Benchmark
from rl_agents.trainer.simulation import Simulation


def main():
    env = prepare_environment()
    evaluate_agent(env)
    # benchmark_agents(env)


def prepare_environment():
    gym.logger.set_level(gym.logger.INFO)
    env = gym.make('highway-v0')
    env.set_difficulty_level("HARD")
    return env


def evaluate_agent(env):
    # agent = TTCVIAgent()
    # agent = dqn_pytorch(env)
    # agent = mcts(env)
    agent = mcts_with_prior(env, temperature=0.1)
    sim = Simulation(env, agent, num_episodes=15000, sim_seed=None, recover=True)
    sim.train()


def benchmark_agents(env):
    # agents = [mcts(env, iterations=20), mcts(env, iterations=60), mcts(env, iterations=100)]
    agents = [mcts_with_prior(env, temperature=0.1),
              mcts_with_prior(env, temperature=0.2),
              mcts_with_prior(env, temperature=0.5),
              mcts_with_prior(env, temperature=1.0),
              mcts_with_prior(env, temperature=2.0)]
    benchmark = Benchmark(env, agents, num_episodes=6)
    benchmark.run()


def dqn_pytorch(environment):
    config = dict(model=dict(type="DuelingNetwork",
                             layers=[512, 512]),
                  memory_capacity=15000/3*20,
                  batch_size=100,
                  gamma=0.9,
                  exploration=dict(method="EpsilonGreedy",
                                   tau=15000/2*15),
                  target_update=30*30)
    agent = DQNPytorchAgent(environment, config)
    print(agent.config)
    return agent


def mcts(environment, iterations=75, temperature=10):
    from functools import partial
    config = dict(iterations=iterations,
                  temperature=temperature,
                  max_depth=7)
    fast_policy = partial(MCTSAgent.preference_policy, action_index=environment.ACTIONS_INDEXES['FASTER'])
    return MCTSAgent(environment,
                     config,
                     prior_policy=fast_policy,
                     rollout_policy=MCTSAgent.random_available_policy)


def mcts_with_prior(environment, temperature=0.5):
    agent_config = dict(model=dict(type="FCNetwork",
                                   layers=[512, 512]),
                        exploration=dict(method="Boltzmann",
                                         temperature=temperature),)
    prior_agent = DQNPytorchAgent(environment, agent_config)
    agent = MCTSWithPriorPolicyAgent(environment,
                                     prior_agent=prior_agent)
    agent.prior_agent.load('out/HighwayEnv/DQNPytorchAgent/saved_models/hard-l512.tar')
    return agent


if __name__ == '__main__':
    num_processes = 1
    if num_processes > 1:
        for i in range(4):
            p = multiprocessing.Process(target=main)
            p.start()
    else:
        main()
