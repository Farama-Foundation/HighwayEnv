from highway_env.agent.mcts import MCTSAgent
from highway_env.road.road import Road
from highway_env.simulation.simulation import Simulation
from highway_env.vehicle.control import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.dynamics import Obstacle

import gym
import highway_env


def main():
    env = gym.make('highway-v0')
    road = Road.create_random_road(lanes_count=2, lane_width=4.0, vehicles_count=0, vehicles_type=IDMVehicle)
    vehicle = MDPVehicle.create_random(road)
    road.vehicles.append(vehicle)
    env.road = road
    env.vehicle = vehicle

    agent = MCTSAgent(env, iterations=100, temperature=20*5)  # compare step by subtree and step by prior
    sim = Simulation(env, agent)

    t = 0
    sim.render()
    while not sim.done:
        sim.step()
        t += 1
        if t == 10:
            print('Added obstacle')
            env.road.vehicles.append(Obstacle(road, [env.vehicle.position[0] + 50., env.vehicle.position[1]]))
    sim.quit()


if __name__ == '__main__':
    main()
