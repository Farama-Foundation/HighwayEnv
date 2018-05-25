from __future__ import division, print_function

import multiprocessing
import glob
import gym

from highway_env.envs.abstract import AbstractEnv
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from rl_agents.agents.tree_search.mcts import MCTSAgent, RobustMCTSAgent
from rl_agents.trainer.analyzer import RunAnalyzer
from rl_agents.trainer.simulation import Simulation


def idle_policy(state):
    return MCTSAgent.preference_policy(state, AbstractEnv.ACTIONS_INDEXES['IDLE'])


def fast_policy(state):
    return MCTSAgent.preference_policy(state, AbstractEnv.ACTIONS_INDEXES['FASTER'])


def assume_linear(env):
    return env.change_vehicles(LinearVehicle)


def assume_idm(env):
    return env.change_vehicles(IDMVehicle)


def evaluate(world_vehicle_type, agent, agent_name):
    gym.logger.set_level(gym.logger.INFO)
    env = gym.make('highway-merge-v0')
    env.other_vehicles_type = world_vehicle_type
    agent.env = env
    directory = 'out/robust_{}_{}'.format(world_vehicle_type.__name__, agent_name)
    sim = Simulation(env, agent, directory=directory, num_episodes=5)
    sim.test()


if __name__ == '__main__':
    jobs = []
    for world_type in [IDMVehicle, LinearVehicle]:
        for (agent_type, name) in \
                [(MCTSAgent(None, iterations=50,
                            env_preprocessor=assume_linear,
                            prior_policy=fast_policy,
                            rollout_policy=idle_policy), LinearVehicle.__name__),
                 (MCTSAgent(None, iterations=50,
                            env_preprocessor=assume_idm,
                            prior_policy=fast_policy,
                            rollout_policy=idle_policy), IDMVehicle.__name__),
                 (RobustMCTSAgent(None,
                                  iterations=50,
                                  models=[assume_linear,
                                          assume_idm],
                                  prior_policy=fast_policy,
                                  rollout_policy=idle_policy), RobustMCTSAgent.__name__)]:
            p = multiprocessing.Process(target=evaluate, args=(world_type, agent_type, name))
            jobs.append(p)
            p.start()
    for job in jobs:
        job.join()

    runs = glob.glob('out/robust*')
    RunAnalyzer('', runs)
