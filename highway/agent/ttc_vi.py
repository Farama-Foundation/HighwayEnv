from __future__ import division, print_function
import numpy as np
from highway.agent.abstract import Agent
from highway import utils


class TTCVIAgent(Agent):
    HORIZON = 10.0
    TIME_QUANTIFICATION = 1.0
    GAMMA = 1.0

    def __init__(self, state):
        self.state = state
        self.grids = np.zeros((state.ego_vehicle.SPEED_COUNT,
                              len(state.ego_vehicle.road.lanes),
                              int(self.HORIZON / self.TIME_QUANTIFICATION)))
        self.V, self.L, self.T = np.shape(self.grids)
        self.value = np.zeros(np.shape(self.grids))
        self.state_reward = np.zeros(np.shape(self.grids))
        self.action_reward = {0: -state.LANE_CHANGE_COST, 1: 0, 2: -state.LANE_CHANGE_COST, 3: 0, 4: 0}
        self.lane = self.speed = None

    def plan(self, state):
        # Update state and reward
        self.state = state
        self.update_ttc_state()
        self.state_reward = \
            - state.COLLISION_COST * self.grids \
            + state.RIGHT_LANE_REWARD * np.tile(np.arange(self.L)[np.newaxis, :, np.newaxis], (self.V, 1, self.T)) \
            + state.HIGH_VELOCITY_REWARD * np.tile(np.arange(self.V)[:, np.newaxis, np.newaxis], (1, self.L, self.T))

        # Run value iteration
        self.value.fill(0)
        self.value_iteration()
        print(self.value)

        # Return chosen trajectory
        path, actions = self.pick_trajectory()
        return actions

    def update_ttc_state(self):
        self.fill_ttc_grid()
        self.lane = self.state.ego_vehicle.lane_index
        self.speed = self.state.ego_vehicle.speed_index()

    def fill_ttc_grid(self):
        self.grids.fill(0)
        for velocity_index in range(self.grids.shape[0]):
            ego_velocity = self.state.ego_vehicle.index_to_speed(velocity_index)
            for other in self.state.ego_vehicle.road.vehicles:
                if (other is self.state.ego_vehicle) or (ego_velocity == other.velocity):
                    continue
                margin = other.LENGTH / 2 + self.state.ego_vehicle.LENGTH / 2
                collision_points = [(0, 1), (-margin, 0.5), (margin, 0.5)]
                for m, cost in collision_points:
                    distance = self.state.ego_vehicle.lane_distance_to(other) + m
                    time_to_collision = distance / utils.not_zero(ego_velocity - other.velocity)
                    if time_to_collision < 0:
                        continue
                    # Quantize time-to-collision to both upper and lower values
                    l = other.lane_index
                    for t in [int(time_to_collision / self.TIME_QUANTIFICATION),
                              int(np.ceil(time_to_collision / self.TIME_QUANTIFICATION))]:
                        if 0 <= l < np.shape(self.grids)[1] and 0 <= t < np.shape(self.grids)[2]:
                            self.grids[velocity_index, l, t] = max(self.grids[velocity_index, l, t], cost)

    def value_iteration(self, steps=50):
        for _ in range(steps):
            self.backup()

    def backup(self):
        new_value = np.zeros(np.shape(self.value))
        for h in range(self.V):
            for i in range(self.L):
                for j in range(self.T):
                    q_values = self.get_q_values(h, i, j)
                    if q_values:
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
        return self.state_reward[h, i, j] + self.action_reward[action]

    def transition_model(self, a, h, i, j):
        """
            Deterministic transition from a position in the grid to the next.
            a: action index
            h: velocity index
            i: lane index
            j: time index
        """
        if a == 0:
            return self.clamp_position(h, i - 1, j + 1)  # LEFT
        elif a == 1:
            return self.clamp_position(h, i, j + 1)  # IDLE
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
        if j == self.T - 1 or self.grids[h, i, j] == 1:
            return q_values  # Terminal state
        for a in range(0, 5):
            o, p, q = self.transition_model(a, h, i, j)
            q_values.append(self.reward(h, i, j, a) + self.value[o, p, q])
        return q_values

    def pick_action(self):
        """
            Pick an optimal action according to the current estimated value function.
        """
        h, i, j = self.speed, self.lane, 0
        q_values = self.get_q_values(h, i, j)
        a = int(np.argmax(q_values))
        return self.state.ACTIONS[a]

    def pick_trajectory(self):
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
            actions.append(self.state.ACTIONS[a])
            h, i, j = self.transition_model(a, h, i, j)
            path.append((h, i, j))
            q_values = self.get_q_values(h, i, j)
        # If terminal state, return default action
        if not actions:
            actions = [self.state.ACTIONS[1]]
        return path, actions
