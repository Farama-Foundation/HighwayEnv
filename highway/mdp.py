from __future__ import division, print_function
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import pygame

from highway.road import Road
from highway import utils


class RoadMDP(object):
    """
        A MDP representing the times to collision between the ego-vehicle and
        other vehicles on the road.
    """
    ACTION_TIMESTEP = 0.1
    MAX_ACTION_DURATION = 1.0

    def __init__(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle

    def step(self, action):
        # Send order to low-level agent
        self.ego_vehicle.act(action)

        # Inner high-frequency loop
        for k in range(int(self.MAX_ACTION_DURATION / self.ACTION_TIMESTEP)):
            self.ego_vehicle.road.step(self.ACTION_TIMESTEP)


class TTCMDP(RoadMDP):
    HORIZON = 10.0
    TIME_QUANTIFICATION = 1.0
    GAMMA = 1.0

    COLLISION_COST = 10
    LANE_CHANGE_COST = 0.00
    LEFT_LANE_COST = -0.01
    HIGH_VELOCITY_REWARD = 0.5

    actions = {0: 'IDLE', 1: 'LANE_LEFT', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}
    cost = {0: 0, 1: -LANE_CHANGE_COST, 2: -LANE_CHANGE_COST, 3: 0, 4: 0}

    def __init__(self, ego_vehicle):
        super(TTCMDP, self).__init__(ego_vehicle)
        self.grids, self.lane, self.speed = self.get_state()
        self.V, self.L, self.T = np.shape(self.grids)
        self.value = np.zeros(np.shape(self.grids))
        self.state_reward = - self.COLLISION_COST * self.grids \
            - self.LEFT_LANE_COST * np.tile(np.arange(self.L)[np.newaxis, :, np.newaxis], (self.V, 1, self.T)) \
            + self.HIGH_VELOCITY_REWARD * np.tile(np.arange(self.V)[:, np.newaxis, np.newaxis], (1, self.L, self.T))

    def get_state(self):
        grids = np.zeros((self.ego_vehicle.SPEED_COUNT,
                          len(self.ego_vehicle.road.lanes),
                          int(self.HORIZON / self.TIME_QUANTIFICATION)))
        self.fill_ttc_grid(grids)
        lane = self.ego_vehicle.lane_index
        speed = self.ego_vehicle.speed_index()
        return grids, lane, speed

    def fill_ttc_grid(self, grids):
        for velocity_index in range(grids.shape[0]):
            ego_velocity = self.ego_vehicle.index_to_speed(velocity_index)
            for other in self.ego_vehicle.road.vehicles:
                if (other is self.ego_vehicle) or (ego_velocity == other.velocity):
                    continue
                margin = other.LENGTH / 2 + self.ego_vehicle.LENGTH / 2
                collision_points = [(0, 2), (-margin, 1), (margin, 1)]
                for m, cost in collision_points:
                    distance = self.ego_vehicle.lane_distance_to(other) + m
                    time_to_collision = distance / utils.not_zero(ego_velocity - other.velocity)
                    if time_to_collision < 0:
                        continue
                    # Quantize time-to-collision to both upper and lower values
                    l = other.lane_index
                    for t in [int(time_to_collision / self.TIME_QUANTIFICATION),
                              int(np.ceil(time_to_collision / self.TIME_QUANTIFICATION))]:
                        if 0 <= l < np.shape(grids)[1] and 0 <= t < np.shape(grids)[2]:
                            grids[velocity_index, l, t] = max(grids[velocity_index, l, t], cost)

    def value_iteration(self, steps=50):
        for _ in range(steps):
            self.update()

    def update(self):
        new_value = np.zeros(np.shape(self.value))
        for h in range(self.V):
            for i in range(self.L):
                for j in range(self.T):
                    q_values = self.get_q_values(h, i, j)
                    if len(q_values):
                        new_value[h, i, j] = self.GAMMA * np.max(q_values)
                    else:
                        new_value[h, i, j] = self.state_reward[h, i, j]
        self.value = new_value

    def clamp_position(self, h, i, j):
        o = min(max(h, 0), np.shape(self.value)[0] - 1)
        p = min(max(i, 0), np.shape(self.value)[1] - 1)
        q = min(max(j, 0), np.shape(self.value)[2] - 1)
        return o, p, q

    def reward(self, h, i, j, action):
        return self.state_reward[h, i, j] + self.cost[action]

    def transition_model(self, a, h, i, j):
        """
            Deterministic transition from a position in the grid to the next.
            a: action index
            h: velocity index
            i: lane index
            j: time index
        """
        if a == 0:
            return self.clamp_position(h, i, j + 1)  # IDLE
        elif a == 1:
            return self.clamp_position(h, i - 1, j + 1)  # LEFT
        elif a == 2:
            return self.clamp_position(h, i + 1, j + 1)  # RIGHT
        elif a == 3:
            return self.clamp_position(h + 1, i, j + 1)  # FASTER
        elif a == 4:
            return self.clamp_position(h - 1, i, j + 1)  # SLOWER
        else:
            return None

    def get_q_values(self, h, i, j):
        q_values = []
        if j == self.T - 1:
            return q_values  # Terminal state
        for k in range(0, 5):
            o, p, q = self.transition_model(k, h, i, j)
            q_values.append(self.reward(h, i, j, k) + self.value[o, p, q])
        return q_values

    def pick_action(self):
        """
            Pick an optimal action according to the current estimated value function.
        """
        h, i, j = self.speed, self.lane, 0
        q_values = self.get_q_values(h, i, j)
        a = int(np.argmax(q_values))
        return self.actions[a]

    def plan(self):
        """
            Get a list of successive states following the optimal policy
            extracted from the current estimated value function.
        """
        h, i, j = self.speed, self.lane, 0
        path = [(h, i, j)]
        actions = []
        q_values = self.get_q_values(h, i, j)
        while len(q_values):
            a = int(np.argmax(q_values))
            actions.append(self.actions[a])
            h, i, j = self.transition_model(a, h, i, j)
            path.append((h, i, j))
            q_values = self.get_q_values(h, i, j)
        return path, actions

    def display(self, surface):
        black = (0, 0, 0)
        red = (255, 0, 0)

        norm = mpl.colors.Normalize(vmin=-30, vmax=20)
        cmap = cm.jet_r
        cell_size = (surface.get_width() // self.T, surface.get_height() // (self.L * self.V))
        velocity_size = surface.get_height() // self.V

        for h in range(self.V):
            for i in range(self.L):
                for j in range(self.T):
                    color = cmap(norm(self.value[h, i, j]), bytes=True)
                    pygame.draw.rect(surface, color, (
                        j * cell_size[0], i * cell_size[1] + h * velocity_size, cell_size[0], cell_size[1]), 0)
            pygame.draw.line(surface, black, (0, h * velocity_size), (self.T * cell_size[0], h * velocity_size), 1)
        path, actions = self.plan()
        for (h, i, j) in path:
            pygame.draw.rect(surface, red,
                             (j * cell_size[0], i * cell_size[1] + h * velocity_size, cell_size[0], cell_size[1]), 1)


def test():
    from highway.simulation import Simulation
    from highway.vehicle import MDPVehicle, LinearVehicle
    road = Road.create_random_road(lanes_count=4, lane_width=4.0, vehicles_count=50, vehicles_type=LinearVehicle)
    sim = Simulation(road, ego_vehicle_type=MDPVehicle)
    sim.RECORD_VIDEO = False
    while not sim.done:
        sim.process()
    sim.quit()


if __name__ == '__main__':
    test()
