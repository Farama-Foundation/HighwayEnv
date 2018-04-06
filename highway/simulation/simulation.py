from __future__ import division, print_function
import numpy as np

from highway.vehicle.control import MDPVehicle
from highway.mdp.road_mdp import RoadMDP
from highway.agent.ttc_vi import TTCVIAgent


class Simulation:
    SIMULATION_FREQUENCY = 15
    POLICY_FREQUENCY = 1
    TRAJECTORY_TIMESTEP = 0.35
    MAXIMUM_SIMULATION_TIME = 3 * 60
    dt = 1 / SIMULATION_FREQUENCY

    def __init__(self, road, ego_vehicle_type=None, agent_type=TTCVIAgent):
        self.road = road

        # Create a new controlled ego-vehicle
        if ego_vehicle_type:
            self.vehicle = ego_vehicle_type.create_random(self.road, 25)
            self.road.vehicles.append(self.vehicle)
        else:
            self.vehicle = None

        # Create a new agent controlling the ego-vehicle
        if agent_type and self.vehicle and isinstance(self.vehicle, MDPVehicle):
            self.agent = agent_type(RoadMDP(self.vehicle))
        else:
            self.agent = None

        self.t = 0
        self.done = False
        self.pause = False
        self.planned_trajectory = None

    def process(self):
        if not self.pause:
            self.act()
            self.step()
            self.ending_criterion()

    def act(self):
        # Default action for all vehicles
        self.road.act()

        # Plan with the agent for the ego-vehicle
        policy_call = self.t % (self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY) == 0
        if self.agent and policy_call:
            mdp_state = RoadMDP(self.vehicle,
                                action_timestep=self.dt,
                                action_duration=1/self.POLICY_FREQUENCY).simplified()
            actions = self.agent.plan(mdp_state)
            self.planned_trajectory = self.vehicle.predict_trajectory(actions,
                                                                      1 / self.POLICY_FREQUENCY,
                                                                      self.TRAJECTORY_TIMESTEP,
                                                                      self.dt)
            if actions:
                self.vehicle.act(actions[0])

    def step(self):
        self.road.step(self.dt)
        self.t += 1

    def ending_criterion(self):
        if self.t/self.SIMULATION_FREQUENCY > self.MAXIMUM_SIMULATION_TIME \
                or (self.vehicle.crashed and self.vehicle.velocity < 1) \
                or ((len(self.road.vehicles) > 1)
                    and (self.vehicle.position[0] > 50 + np.max(
                        [o.position[0] for o in self.road.vehicles if o is not self.vehicle]))):
            self.done = True
