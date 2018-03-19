from __future__ import division, print_function
import numpy as np

from highway.agent.mcts import MCTSAgent
from highway.road.lane import LineType, StraightLane, SineLane, LanesConcatenation
from highway.road.road import Road
from highway.mdp.abstract import MDP
from highway.mdp.road_mdp import RoadMDP
from highway.simulation import Simulation
from highway.vehicle.behavior import IDMVehicle
from highway.vehicle.control import ControlledVehicle, MDPVehicle
from highway.vehicle.dynamics import Obstacle


class InsertionMDP(MDP):
    """
        Describe an MDP with a particular lanes and vehicle configuration, and a specific reward function.
    """
    VELOCITY_REWARD = 1.0
    RIGHT_LANE_REWARD = 0.0
    ACCELERATION_COST = 0
    LANE_CHANGE_COST = 0

    def __init__(self):
        self.road = InsertionMDP.make_road()
        self.ego_vehicle, self.inserting_v = InsertionMDP.make_vehicles(self.road)
        self.road_mdp = RoadMDP(self.ego_vehicle)
        self.ACTIONS = self.road_mdp.ACTIONS

    def step(self, action):
        self.road_mdp.step(action)
        return self.reward(action)

    def reward(self, action):
        action_reward = {0: -self.LANE_CHANGE_COST, 1: 0, 2: -self.LANE_CHANGE_COST, 3: -self.ACCELERATION_COST, 4: -self.ACCELERATION_COST}
        reward = -RoadMDP.COLLISION_COST * self.ego_vehicle.crashed \
            + self.RIGHT_LANE_REWARD * (self.ego_vehicle.lane_index == len(self.road.lanes)-2) \
            + self.VELOCITY_REWARD * self.ego_vehicle.velocity_index

        reward += 0.1*self.inserting_v.velocity
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
        road = Road([l1, l2])
        road.vehicles.append(Obstacle(road, lc2.position(ends[2], 0)))
        return road

    @staticmethod
    def make_vehicles(road):
        ego_vehicle = MDPVehicle(road, road.lanes[-2].position(-40, 0), velocity=30)
        road.vehicles.append(ego_vehicle)

        road.vehicles.append(ControlledVehicle(road, road.lanes[-2].position(20, 0), velocity=30))
        inserting_v = IDMVehicle(road, road.lanes[-1].position(60, 0), velocity=20)
        inserting_v.target_velocity = 28
        road.vehicles.append(inserting_v)
        return ego_vehicle, inserting_v


def run():
    np.random.seed(1)
    mdp = InsertionMDP()
    sim = Simulation(mdp.road)
    sim.vehicle = mdp.ego_vehicle
    sim.agent = MCTSAgent(mdp, prior_policy=MCTSAgent.fast_policy, rollout_policy=MCTSAgent.idle_policy, iterations=50)
    sim.RECORD_VIDEO = False
    sim.road_surface.centering_position = 0.5

    while not sim.done:
        sim.handle_events()

        # Default action for all vehicles
        sim.road.act()

        # Planning for ego-vehicle
        policy_call = sim.t % (sim.FPS // (sim.REAL_TIME_RATIO * sim.POLICY_FREQUENCY)) == 0
        if sim.agent and policy_call:
            actions = sim.agent.plan(mdp)
            # sim.display_prediction()
            sim.trajectory = sim.vehicle.predict_trajectory(actions,
                                                              RoadMDP.MAX_ACTION_DURATION,
                                                              sim.TRAJECTORY_TIMESTEP,
                                                              sim.dt)
            sim.vehicle.act(actions[0])
            print("reward", mdp.reward(mdp.road_mdp.ACTIONS_INDEXES[actions[0]]))

        sim.step()
        sim.display()
    sim.quit()


if __name__ == '__main__':
    np.random.seed(1)
    run()
