from __future__ import division, print_function
import numpy as np
import copy

from highway.agent.mcts import MCTSAgent
from highway.road.lane import LineType, StraightLane, SineLane, LanesConcatenation
from highway.road.road import Road
from highway.mdp.abstract import MDP
from highway.mdp.road_mdp import RoadMDP
from highway.simulation.graphics import SimulationWindow
from highway.simulation.simulation import Simulation
from highway.vehicle.behavior import IDMVehicle, LinearVehicle
from highway.vehicle.control import ControlledVehicle, MDPVehicle
from highway.vehicle.dynamics import Obstacle


class MergeMDP(MDP):
    """
        Describe an MDP with a particular lanes and vehicle configuration, and a specific reward function.
    """
    VELOCITY_REWARD = 1.0
    MERGING_VELOCITY_REWARD = 2.0 / 20.0
    RIGHT_LANE_REWARD = 0.5
    ACCELERATION_COST = 0
    LANE_CHANGE_COST = 0*1.0

    def __init__(self):
        road = MergeMDP.make_road()
        ego_vehicle = MergeMDP.make_vehicles(road)
        self.road_mdp = RoadMDP(ego_vehicle)
        self.ACTIONS = self.road_mdp.ACTIONS

    def step(self, action):
        self.road_mdp.step(action)
        return self.reward(action)

    def reward(self, action):
        action_reward = {0: -self.LANE_CHANGE_COST, 1: 0, 2: -self.LANE_CHANGE_COST, 3: -self.ACCELERATION_COST, 4: -self.ACCELERATION_COST}
        reward = -RoadMDP.COLLISION_COST * self.road_mdp.ego_vehicle.crashed \
            + self.RIGHT_LANE_REWARD * self.road_mdp.ego_vehicle.lane_index \
            + self.VELOCITY_REWARD * self.road_mdp.ego_vehicle.velocity_index

        road = self.road_mdp.ego_vehicle.road
        for vehicle in road.vehicles:
            if vehicle.lane_index == len(road.lanes)-1 and isinstance(vehicle, ControlledVehicle):
                reward -= self.MERGING_VELOCITY_REWARD * (vehicle.target_velocity - vehicle.velocity)
        return reward + action_reward[action]

    @classmethod
    def get_actions(cls):
        return RoadMDP.road_mdp.get_actions()

    def get_available_actions(self):
        return self.road_mdp.get_available_actions()

    def is_terminal(self):
        return self.road_mdp.is_terminal()

    @staticmethod
    def make_road():
        ends = [80, 80, 80]
        l0 = StraightLane(np.array([0, 0]), 0, 4.0, [LineType.CONTINUOUS, LineType.NONE])
        l1 = StraightLane(np.array([0, 4]), 0, 4.0, [LineType.STRIPED, LineType.CONTINUOUS])

        lc0 = StraightLane(np.array([0, 6.5 + 4 + 4]), 0, 4.0,
                           [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[-np.inf, ends[0]], forbidden=True)
        amplitude = 3.3
        lc1 = SineLane(lc0.position(ends[0], -amplitude), 0, 4.0, amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2,
                       [LineType.STRIPED, LineType.STRIPED], bounds=[0, ends[1]], forbidden=True)
        lc2 = StraightLane(lc1.position(ends[1], 0), 0, 4.0,
                           [LineType.NONE, LineType.CONTINUOUS], bounds=[0, ends[2]], forbidden=True)
        l2 = LanesConcatenation([lc0, lc1, lc2])
        road = Road([l0, l1, l2])
        road.vehicles.append(Obstacle(road, lc2.position(ends[2], 0)))
        return road

    @staticmethod
    def make_vehicles(road):
        ego_vehicle = MDPVehicle(road, road.lanes[-2].position(-40, 0), velocity=30)
        road.vehicles.append(ego_vehicle)

        road.vehicles.append(IDMVehicle(road, road.lanes[0].position(20, 0), velocity=30))
        road.vehicles.append(IDMVehicle(road, road.lanes[1].position(35, 0), velocity=30))
        road.vehicles.append(IDMVehicle(road, road.lanes[0].position(-65, 0), velocity=31.5))

        IDMVehicle.TIME_WANTED = 1.0
        IDMVehicle.POLITENESS = 0.0
        merging_v = IDMVehicle(road, road.lanes[-1].position(70, 0), velocity=20)
        merging_v.target_velocity = 30
        road.vehicles.append(merging_v)
        return ego_vehicle

    def change_agents_to(self, agent_type):
        """
            Change the type of all agents on the road
        :param agent_type: The new type of agents
        :return: a new RoadMDP with modified agents type
        """
        state_copy = copy.deepcopy(self)
        state_copy.road_mdp = state_copy.road_mdp.change_agents_to(agent_type)
        return state_copy


def run():
    mdp = MergeMDP()
    sim = Simulation(mdp.road_mdp.ego_vehicle.road)
    sim.vehicle = mdp.road_mdp.ego_vehicle
    sim.agent = MCTSAgent(mdp,
                          prior_policy=MCTSAgent.fast_policy,
                          rollout_policy=MCTSAgent.idle_policy,
                          iterations=100,
                          assume_vehicle_type=LinearVehicle)
    window = SimulationWindow(sim)
    # sim.agent = SingleTrajectoryAgent(['LANE_LEFT'], 'IDLE')

    action = None
    while not window.done:
        window.handle_events()

        # Default action for all vehicles
        sim.road.act()

        # Planning for ego-vehicle
        policy_call = sim.t % (sim.SIMULATION_FREQUENCY // sim.POLICY_FREQUENCY) == 0
        if sim.agent and policy_call:
            if action:
                print('reward', mdp.reward(RoadMDP.ACTIONS_INDEXES[action]))
            actions = sim.agent.plan(mdp)  # Here the mdp must be merge mdp and not created on the fly
            sim.planned_trajectory = sim.vehicle.predict_trajectory(actions,
                                                                    1 / sim.POLICY_FREQUENCY,
                                                                    sim.TRAJECTORY_TIMESTEP,
                                                                    sim.dt)
            action = actions[0]
            sim.vehicle.act(action)
            print('action', action)

        # End of episode
        if sim.vehicle.position[0] > 400:
            sim.done = True

        sim.step()
        window.display()
    window.quit()


if __name__ == '__main__':
    # np.random.seed(3)
    run()
