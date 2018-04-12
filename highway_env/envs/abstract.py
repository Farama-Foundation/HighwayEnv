from __future__ import division, print_function
import copy
import gym
from gym import spaces
import numpy as np

from highway_env.envs.graphics import EnvViewer


class AbstractEnv(gym.Env):
    """
        A generic environment for various tasks involving a vehicle driving on a road.

        The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
        velocity. The action space is fixed, but the observation space and reward function must be defined in the
        environment implementations.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    ACTIONS = {0: 'LANE_LEFT',
               1: 'IDLE',
               2: 'LANE_RIGHT',
               3: 'FASTER',
               4: 'SLOWER'}
    """
        A mapping of action indexes to action labels
    """
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}
    """
        A mapping of action labels to action indexes
    """

    SIMULATION_FREQUENCY = 15
    """
        The frequency at which the system dynamics are simulated [Hz]
    """
    POLICY_FREQUENCY = 1
    """
        The frequency at which the agent can take actions [Hz]
    """
    PERCEPTION_DISTANCE = 150
    """
        The maximum distance of any vehicle present in the observation [m]
    """

    def __init__(self, road, vehicle):
        self.road = road
        self.vehicle = vehicle

        self.done = False
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.observation_space = None
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)

    def _observation(self):
        """
            Return the observation of the current state, which must be consistent with self.observation_space.
        :return: the observation
        """
        raise NotImplementedError()

    def _reward(self, action):
        """
            Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError()

    def _is_terminal(self):
        """
            Check whether the current state is a terminal state
        :return:is the state terminal
        """
        raise NotImplementedError()

    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        raise NotImplementedError()

    def step(self, action):
        """
            Perform an action and step the environment dynamics.

            The action is executed by the ego-vehicle, and all other vehicles on the road performs their default
            behaviour for several simulation timesteps until the next decision making step.
        :param int action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        # Forward action to the vehicle
        self.vehicle.act(self.ACTIONS[action])

        # Simulate
        for k in range(int(self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY)):
            self.road.act()
            self.road.step(1 / self.SIMULATION_FREQUENCY)

            # Automatically render intermediate simulation steps if a viewer has been launched
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break

        obs = self._observation()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = {}

        return obs, reward, terminal, info

    def render(self, mode='human'):
        """
            Render the environment.

            Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self, record_video=False)

        # If the frame has already been rendered, do nothing
        if self.should_update_rendering:
            self.viewer.display()

        if mode == 'rgb_array':
            return self.viewer.get_image()
        elif mode == 'human':
            self.viewer.handle_events()
        self.should_update_rendering = False

    def close(self):
        """
            Close the environment.

            Will close the environment viewer if it exists.
        """
        self.done = True
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

    def _automatic_rendering(self):
        """
            Automatically render the intermediate frames while an action is still ongoing.
            This allows to render the whole video and not only single steps corresponding to agent decision-making.

            If a callback has been set, use it to perform the rendering. This is useful for the environment wrappers
            such as video-recording monitor that need to access these intermediate renderings.
        """
        if self.viewer is not None:
            self.should_update_rendering = True

            if self.automatic_rendering_callback:
                self.automatic_rendering_callback()
            else:
                self.render(self.rendering_mode)

    def _simplified(self):
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
        """
            Perform a deep copy but without copying the environment viewer.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', 'automatic_rendering_callback']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result
