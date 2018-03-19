from __future__ import division, print_function
import copy

from highway.mdp.abstract import MDP


class RoadMDP(MDP):
    """
        A MDP representing the times to collision between the ego-vehicle and
        other vehicles on the road.
    """
    ACTION_TIMESTEP = 1/30
    MAX_ACTION_DURATION = 1.0

    ACTIONS = {0: 'LANE_LEFT',
               1: 'IDLE',
               2: 'LANE_RIGHT',
               3: 'FASTER',
               4: 'SLOWER'}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    COLLISION_COST = 10
    LANE_CHANGE_COST = 1.0
    RIGHT_LANE_REWARD = 0.5
    HIGH_VELOCITY_REWARD = 1.0

    SAFE_DISTANCE = 150

    def __init__(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle

    def step(self, action):
        # Send order to low-level agent
        self.ego_vehicle.act(self.ACTIONS[action])

        # Inner high-frequency loop
        for k in range(int(self.MAX_ACTION_DURATION / self.ACTION_TIMESTEP)):
            self.ego_vehicle.road.act()
            self.ego_vehicle.road.step(self.ACTION_TIMESTEP)

        return self.reward(action)

    @classmethod
    def get_actions(cls):
        """
            Get the action space.
        """
        return cls.ACTIONS

    def get_available_actions(self):
        """
            Get the list of currently available actions.

            Lane changes are not available on the boundary of the road, and velocity changes are not available at
            maximal or minimal velocity.

        :return: the list of available actions
        """
        actions = [self.ACTIONS_INDEXES['IDLE']]
        li = self.ego_vehicle.lane_index
        if li > 0 \
                and self.ego_vehicle.road.lanes[li-1].is_reachable_from(self.ego_vehicle.position):
            actions.append(self.ACTIONS_INDEXES['LANE_LEFT'])
        if li < len(self.ego_vehicle.road.lanes) - 1 \
                and self.ego_vehicle.road.lanes[li+1].is_reachable_from(self.ego_vehicle.position):
            actions.append(self.ACTIONS_INDEXES['LANE_RIGHT'])
        if self.ego_vehicle.velocity_index < self.ego_vehicle.SPEED_COUNT - 1:
            actions.append(self.ACTIONS_INDEXES['FASTER'])
        if self.ego_vehicle.velocity_index > 0:
            actions.append(self.ACTIONS_INDEXES['SLOWER'])
        return actions

    def reward(self, action):
        """
            Define the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        action_reward = {0: -self.LANE_CHANGE_COST, 1: 0, 2: -self.LANE_CHANGE_COST, 3: 0, 4: 0}
        state_reward = \
            - self.COLLISION_COST*self.ego_vehicle.crashed \
            + self.RIGHT_LANE_REWARD*self.ego_vehicle.lane_index \
            + self.HIGH_VELOCITY_REWARD*self.ego_vehicle.speed_index()
        return action_reward[action]+state_reward

    def simplified(self):
        """
            Return a simplified copy of the state where distant vehicles have been removed from the road.

            This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified mdp state
        """
        state_copy = copy.deepcopy(self)
        ev = state_copy.ego_vehicle
        close_vehicles = []
        for v in ev.road.vehicles:
            if -self.SAFE_DISTANCE/2 < ev.lane_distance_to(v) < self.SAFE_DISTANCE:
                close_vehicles.append(v)
        ev.road.vehicles = close_vehicles
        return state_copy

    def is_terminal(self):
        """
        :return: Whether the current state is a terminal state
        """
        return self.ego_vehicle.crashed
