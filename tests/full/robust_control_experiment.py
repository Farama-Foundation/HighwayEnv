from __future__ import division, print_function

import multiprocessing
import glob
import gym
import highway_env

from highway_env.agent.mcts import MCTSAgent, RobustMCTSAgent
from highway_env.wrappers.analyzer import RunAnalyzer
from highway_env.vehicle.behavior import AggressiveVehicle, DefensiveVehicle, IDMVehicle, LinearVehicle
from highway_env.wrappers.simulation import Simulation
from highway_env.wrappers.monitor import MonitorV2


def evaluate(world_vehicle_type, agent, agent_name):
    env = gym.make('highway-merge-v0')
    env.other_vehicles_type = world_vehicle_type
    directory = 'out/robust_{}_{}'.format(world_vehicle_type.__name__, agent_name)
    monitor = MonitorV2(env, directory, add_subdirectory=False)

    sim = Simulation(monitor, agent, highway_env=env, episodes=5)
    sim.run()


if __name__ == '__main__':
    jobs = []
    for world_type in [AggressiveVehicle, DefensiveVehicle]:
        for (agent_type, name) in [(MCTSAgent(iterations=50, assume_vehicle_type=AggressiveVehicle), AggressiveVehicle.__name__),
                                   (MCTSAgent(iterations=50, assume_vehicle_type=DefensiveVehicle), DefensiveVehicle.__name__),
                                   (RobustMCTSAgent(models=[AggressiveVehicle, DefensiveVehicle], iterations=50), RobustMCTSAgent.__name__)]:
                p = multiprocessing.Process(target=evaluate, args=(world_type, agent_type, name))
                jobs.append(p)
                p.start()
    for job in jobs:
        job.join()

    runs = glob.glob('out/robust*')
    RunAnalyzer('', runs)
