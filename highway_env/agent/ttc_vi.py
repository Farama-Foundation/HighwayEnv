from __future__ import division, print_function
import numpy as np
import operator
from highway_env.agent.abstract import AbstractAgent
from highway_env import utils


class TTCVIAgent(AbstractAgent):
    """
        Implementation of Value Iteration over a Time-To-Collision (TTC) representation of the state.

        The state reward is defined from a occupancy grid over different TTCs and lanes. The grid cells encode the
        probability that the ego-vehicle will collide with another vehicle if it is located on a given lane in a given
        duration, under the hypothesis that every vehicles observed will maintain a constant velocity (including the
        ego-vehicle) and not change lane (excluding the ego-vehicle).

        For instance, in a three-lane road with a vehicle on the left lane with collision predicted in 5s the grid will
        be:
        [0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0]
        The TTC-state is a coordinate (lane, time) within this grid.

        If the ego-vehicle has the ability to change its velocity, an additional layer is added to the occupancy grid
        to iterate over the different velocity choices available.
    """
    HORIZON = 10.0
    """
        The time horizon used in the state representation [s]
    """
    TIME_QUANTIZATION = 1.0
    """
        The time quantization used in the state representation [s]
    """
    GAMMA = 1.0
    """
        The discount factor used for planning, in [0, 1].
    """

    def __init__(self):
        """
            New instance of TTCVI agent.
        """
        self.state = None
        self.grids = self.value = None
        self.V = self.L = self.T = None
        self.state_reward = None
        self.action_reward = self.state_reward = None
        self.lane = self.speed = None

    def plan(self, state):
        """
            Perform a value iteration and return the sequence of optimal actions.

            Compute the TTC-grid to build the associated reward function, and run the value iteration.

        :param state: the current MDP state
        :return: a list of optimal actions
        """
        # Initialize variables if needed
        if self.grids is None:
            self.grids = np.zeros((state.vehicle.SPEED_COUNT,
                                   len(state.vehicle.road.lanes),
                                   int(self.HORIZON / self.TIME_QUANTIZATION)))
            self.V, self.L, self.T = np.shape(self.grids)
            self.value = np.zeros(np.shape(self.grids))

        # Update state and reward
        self.state = state
        self.update_ttc_state()
        self.action_reward = {0: -state.LANE_CHANGE_COST,
                              1: 0,
                              2: -state.LANE_CHANGE_COST,
                              3: 0,
                              4: 0}
        self.state_reward = \
            - state.COLLISION_COST * self.grids \
            + state.RIGHT_LANE_REWARD * np.tile(np.arange(self.L)[np.newaxis, :, np.newaxis], (self.V, 1, self.T)) \
            + state.HIGH_VELOCITY_REWARD * np.tile(np.arange(self.V)[:, np.newaxis, np.newaxis], (1, self.L, self.T))

        # Run value iteration
        self.value.fill(0)
        self.value_iteration()

        # Return chosen trajectory
        path, actions = self.pick_trajectory()
        return actions

    def update_ttc_state(self):
        """
            Extract the TTC-grid and TTC-state (velocity, lane, time=0) from the current MDP state.
        """
        self.fill_ttc_grid()
        self.lane = self.state.vehicle.lane_index
        self.speed = self.state.vehicle.speed_index()

    def fill_ttc_grid(self):
        """
            For each ego-velocity and lane, compute the predicted time-to-collision to each vehicle within the lane and
            store the results in an occupancy grid.
        """
        self.grids.fill(0)
        for velocity_index in range(self.grids.shape[0]):
            ego_velocity = self.state.vehicle.index_to_speed(velocity_index)
            for other in self.state.road.vehicles:
                if (other is self.state.vehicle) or (ego_velocity == other.velocity):
                    continue
                margin = other.LENGTH / 2 + self.state.vehicle.LENGTH / 2
                collision_points = [(0, 1), (-margin, 0.5), (margin, 0.5)]
                for m, cost in collision_points:
                    distance = self.state.vehicle.lane_distance_to(other) + m
                    time_to_collision = distance / utils.not_zero(ego_velocity - other.velocity)
                    if time_to_collision < 0:
                        continue
                    # Quantize time-to-collision to both upper and lower values
                    lane = other.lane_index
                    for time in [int(time_to_collision / self.TIME_QUANTIZATION),
                                 int(np.ceil(time_to_collision / self.TIME_QUANTIZATION))]:
                        if 0 <= lane < np.shape(self.grids)[1] and 0 <= time < np.shape(self.grids)[2]:
                            self.grids[velocity_index, lane, time] = max(self.grids[velocity_index, lane, time], cost)

    def value_iteration(self, steps=50):
        """
            Perform a value iteration over the TTC-state and reward.

        :param steps: number of backup operations.
        """
        for _ in range(steps):
            self.backup()

    def backup(self):
        """
            Apply the Bellman optimal operator to the estimated value function.
        """
        new_value = np.zeros(np.shape(self.value))
        for h in range(self.V):
            for i in range(self.L):
                for j in range(self.T):
                    q_values = self.get_q_values(h, i, j)
                    if q_values:
                        new_value[h, i, j] = self.GAMMA * np.max(list(q_values.values()))
                    else:
                        new_value[h, i, j] = self.state_reward[h, i, j]
        self.value = new_value

    def clip_position(self, h, i, j):
        """
            Clip a position in the TTC grid, so that it stays within bounds.

        :param h: velocity index
        :param i: lane index
        :param j: time index
        :return: the clipped position
        """
        o = min(max(h, 0), np.shape(self.value)[0] - 1)
        p = min(max(i, 0), np.shape(self.value)[1] - 1)
        q = min(max(j, 0), np.shape(self.value)[2] - 1)
        return o, p, q

    def reward(self, h, i, j, action):
        """
            The reward obtained by performing a given action and ending-up in a given TTC-state (h,i,j).

        :param h: velocity index
        :param i: lane index
        :param j: time index
        :param action: the performed action
        :return: the transition reward
        """
        return self.state_reward[h, i, j] + self.action_reward[action]

    def transition_model(self, a, h, i, j):
        """
            Deterministic transition from a position in the grid to the next.

        :param a: action index
        :param h: velocity index
        :param i: lane index
        :param j: time index
        """
        if j == self.T - 1 or self.grids[h, i, j] == 1:
            return None  # Terminal state

        if a == 0:
            return self.clip_position(h, i - 1, j + 1)  # LEFT
        elif a == 1:
            return self.clip_position(h, i, j + 1)  # IDLE
        elif a == 2:
            return self.clip_position(h, i + 1, j + 1)  # RIGHT
        elif a == 3:
            if j == 0:
                return self.clip_position(h + 1, i, j + 1)  # FASTER
            else:
                # FASTER can only be used at the first timestep
                return None
        elif a == 4:
            if j == 0:
                return self.clip_position(h - 1, i, j + 1)  # SLOWER
            else:
                # SLOWER can only be used at the first timestep
                return None
        else:
            return None

    def get_q_values(self, h, i, j):
        """
            Compute the Q-values of all the different actions in a given state, from the state-value function.

        :param h: velocity index
        :param i: lane index
        :param j: time index
        :return: the list of state-action Q-values
        """
        q_values = {}

        for a in range(0, 5):
            next_state = self.transition_model(a, h, i, j)
            if next_state:
                o, p, q = next_state
                q_values[a] = (self.reward(h, i, j, a) + self.value[o, p, q])
        return q_values

    def pick_action(self):
        """
            Pick an optimal action according to the current estimated value function.

        :return: the optimal action
        """
        h, i, j = self.speed, self.lane, 0
        q_values = self.get_q_values(h, i, j)
        a = max(q_values.items(), key=operator.itemgetter(1))[0]
        return a

    def pick_trajectory(self):
        """
            Get a sequence of successive states and actions following the optimal policy extracted from the current
            estimated value function.

        :return: a list of next states, and corresponding actions
        """
        h, i, j = self.speed, self.lane, 0
        path = [(h, i, j)]
        actions = []
        q_values = self.get_q_values(h, i, j)
        while len(q_values):
            a = max(q_values.items(), key=operator.itemgetter(1))[0]
            actions.append(a)
            h, i, j = self.transition_model(a, h, i, j)
            path.append((h, i, j))
            q_values = self.get_q_values(h, i, j)
        # If terminal state, return default action
        if not actions:
            actions = [1]
        return path, actions
