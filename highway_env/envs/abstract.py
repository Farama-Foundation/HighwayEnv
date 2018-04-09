from __future__ import division, print_function
import copy
import gym
from gym import spaces
import numpy as np

from highway_env.envs.graphics import EnvViewer


class AbstractEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    ACTIONS = {0: 'LANE_LEFT',
               1: 'IDLE',
               2: 'LANE_RIGHT',
               3: 'FASTER',
               4: 'SLOWER'}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    SIMULATION_FREQUENCY = 15
    POLICY_FREQUENCY = 1
    PERCEPTION_DISTANCE = 150

    def __init__(self, road, vehicle):
        self.road = road
        self.vehicle = vehicle

        self.viewer = None
        self.observation_space = None
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)

    def observation(self):
        raise NotImplementedError()

    def reward(self, action):
        """
            Define the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError()

    def is_terminal(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        # Forward action to the vehicle
        self.vehicle.act(self.ACTIONS[int(action)])

        # Simulate
        for k in range(int(self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY)):
            self.road.act()
            self.road.step(1 / self.SIMULATION_FREQUENCY)

            # Render simulation
            if self.viewer is not None:
                self.render()

        obs = self.observation()
        reward = self.reward(action)
        done = self.is_terminal()

        return obs, reward, done, {}

    def render(self, mode='human'):
        if mode == 'rgb_array':
            raise NotImplementedError()
        elif mode == 'human':
            self._get_viewer().handle_events()
            self._get_viewer().display()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = EnvViewer(self, record_video=False)
        return self.viewer

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def get_available_actions(self):
        """
            Get the list of currently available actions.

            Lane changes are not available on the boundary of the road, and velocity changes are not available at
            maximal or minimal velocity.

        :return: the list of available actions
        """
        actions = [self.ACTIONS_INDEXES['IDLE']]
        li = self.vehicle.lane_index
        if li > 0 \
                and self.road.lanes[li-1].is_reachable_from(self.vehicle.position):
            actions.append(self.ACTIONS_INDEXES['LANE_LEFT'])
        if li < len(self.road.lanes) - 1 \
                and self.road.lanes[li+1].is_reachable_from(self.vehicle.position):
            actions.append(self.ACTIONS_INDEXES['LANE_RIGHT'])
        if self.vehicle.velocity_index < self.vehicle.SPEED_COUNT - 1:
            actions.append(self.ACTIONS_INDEXES['FASTER'])
        if self.vehicle.velocity_index > 0:
            actions.append(self.ACTIONS_INDEXES['SLOWER'])
        return actions

    # def change_agents_to(self, agent_type):
    #     """
    #         Change the type of all agents on the road
    #     :param agent_type: The new type of agents
    #     :return: a new RoadMDP with modified agents type
    #     """
    #     state_copy = copy.deepcopy(self)
    #     vehicles = state_copy.ego_vehicle.road.vehicles
    #     for i, v in enumerate(vehicles):
    #         if v is not state_copy.ego_vehicle and not isinstance(v, Obstacle):
    #             vehicles[i] = agent_type.create_from(v)
    #     return state_copy

    def simplified(self):
        """
            Return a simplified copy of the environment where distant vehicles have been removed from the road.

            This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        ev = state_copy.vehicle
        close_vehicles = []
        for v in state_copy.road.vehicles:
            if -self.PERCEPTION_DISTANCE/2 < ev.lane_distance_to(v) < self.PERCEPTION_DISTANCE:
                close_vehicles.append(v)
        state_copy.road.vehicles = close_vehicles
        return state_copy

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'viewer':
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result
