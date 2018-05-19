from __future__ import division, print_function

import multiprocessing
import glob
import gym

from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from rl_agents.agents.tree_search.mcts import MCTSAgent, RobustMCTSAgent
from rl_agents.trainer.analyzer import RunAnalyzer
from rl_agents.trainer.simulation import Simulation


def evaluate(world_vehicle_type, agent, agent_name):
    env = gym.make('highway-merge-v0')
    env.other_vehicles_type = world_vehicle_type
    directory = 'out/robust_{}_{}'.format(world_vehicle_type.__name__, agent_name)
    sim = Simulation(env, agent, directory=directory, num_episodes=5)
    sim.test()


if __name__ == '__main__':
    jobs = []
    for world_type in [IDMVehicle, LinearVehicle]:
        for (agent_type, name) in \
                [(MCTSAgent(iterations=50,
                            assume_vehicle_type=LinearVehicle,
                            rollout_policy=MCTSAgent.idle_policy), LinearVehicle.__name__),
                 (MCTSAgent(iterations=50,
                            assume_vehicle_type=IDMVehicle,
                            rollout_policy=MCTSAgent.idle_policy), IDMVehicle.__name__),
                 (RobustMCTSAgent(iterations=50,
                                  models=[LinearVehicle, IDMVehicle],
                                  rollout_policy=MCTSAgent.idle_policy), RobustMCTSAgent.__name__)]:
            p = multiprocessing.Process(target=evaluate, args=(world_type, agent_type, name))
            jobs.append(p)
            p.start()
    for job in jobs:
        job.join()

    runs = glob.glob('out/robust*')
    RunAnalyzer('', runs)
