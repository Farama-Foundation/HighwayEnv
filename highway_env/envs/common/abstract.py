import copy
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from highway_env import utils
from highway_env.envs.common.observation import observation_factory
from highway_env.envs.common.finite_mdp import finite_mdp
from highway_env.envs.common.graphics import EnvViewer
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.control import MDPVehicle
from highway_env.vehicle.kinematics import Obstacle


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

    PERCEPTION_DISTANCE = 6.0 * MDPVehicle.SPEED_MAX
    """
        The maximum distance of any vehicle present in the observation [m]
    """

    STEERING_RANGE = np.pi / 4
    ACCELERATION_RANGE = 5.0

    def __init__(self, config=None):
        # Configuration
        self.config = self.default_config()
        if config:
            self.config.update(config)

        # Seeding
        self.np_random = None
        self.seed()

        # Scene
        self.road = None
        self.vehicle = None

        # Spaces
        self.observation = None
        self.action_space = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.offscreen = self.config.get("offscreen_rendering", False)
        self.enable_auto_render = False

        self.reset()

    @classmethod
    def default_config(cls):
        """
            Default environment configuration.

            Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            "observation": {
                "type": "TimeToCollision"
            },
            "action": {
                "type": "Discrete"
            },
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,  # [px]
            "screen_height": 150,  # [px]
            "centering_position": [0.3, 0.5],
            "show_trajectories": False
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def configure(self, config):
        if config:
            self.config.update(config)

    def define_spaces(self):
        self.observation = observation_factory(self, self.config["observation"])
        self.observation_space = self.observation.space()

        if self.config["action"]["type"] == "Discrete":
            self.action_space = spaces.Discrete(len(self.ACTIONS))
        elif self.config["action"]["type"] == "Continuous":
            self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)

    def _reward(self, action):
        """
            Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError

    def _is_terminal(self):
        """
            Check whether the current state is a terminal state
        :return:is the state terminal
        """
        raise NotImplementedError

    def _cost(self, action):
        """
            A constraint metric, for budgeted MDP.

            If a constraint is defined, it must be used with an alternate reward that doesn't contain it
            as a penalty.
        :param action: the last action performed
        :return: the constraint signal, the alternate (constraint-free) reward
        """
        raise NotImplementedError

    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        self.time = 0
        self.done = False
        self.define_spaces()
        return self.observation.observe()

    def step(self, action):
        """
            Perform an action and step the environment dynamics.

            The action is executed by the ego-vehicle, and all other vehicles on the road performs their default
            behaviour for several simulation timesteps until the next decision making step.
        :param int action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self._simulate(action)

        obs = self.observation.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()

        info = {
            "velocity": self.vehicle.velocity,
            "crashed": self.vehicle.crashed,
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass

        return obs, reward, terminal, info

    def _simulate(self, action=None):
        """
            Perform several steps of simulation with constant action
        """
        for k in range(int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"])):
            if action is not None and \
                    self.time % int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"]) == 0:
                # Forward action to the vehicle
                if self.config["action"]["type"] == "Discrete":
                    self.vehicle.act(self.ACTIONS[action])
                elif self.config["action"]["type"] == "Continuous":
                    self.vehicle.act({
                        "acceleration": action[0] * self.ACCELERATION_RANGE,
                        "steering": action[1] * self.STEERING_RANGE
                    })

            self.road.act()
            self.road.step(1 / self.SIMULATION_FREQUENCY)
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False

    def render(self, mode='human'):
        """
            Render the environment.

            Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self, offscreen=self.offscreen)

        self.enable_auto_render = not self.offscreen

        # If the frame has already been rendered, do nothing
        if self.should_update_rendering:
            self.viewer.display()

        if mode == 'rgb_array':
            image = self.viewer.get_image()
            if not self.viewer.offscreen:
                self.viewer.handle_events()
            self.viewer.handle_events()
            return image
        elif mode == 'human':
            if not self.viewer.offscreen:
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
        for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
            if l_index[2] < self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position):
                actions.append(self.ACTIONS_INDEXES['LANE_LEFT'])
            if l_index[2] > self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position):
                actions.append(self.ACTIONS_INDEXES['LANE_RIGHT'])
        if self.vehicle.velocity_index < self.vehicle.SPEED_COUNT - 1:
            actions.append(self.ACTIONS_INDEXES['FASTER'])
        if self.vehicle.velocity_index > 0:
            actions.append(self.ACTIONS_INDEXES['SLOWER'])
        return actions

    def _automatic_rendering(self):
        """
            Automatically render the intermediate frames while an action is still ongoing.
            This allows to render the whole video and not only single steps corresponding to agent decision-making.

            If a callback has been set, use it to perform the rendering. This is useful for the environment wrappers
            such as video-recording monitor that need to access these intermediate renderings.
        """
        if self.viewer is not None and self.enable_auto_render:
            self.should_update_rendering = True

            if self.automatic_rendering_callback:
                self.automatic_rendering_callback()
            else:
                self.render(self.rendering_mode)

    def simplify(self):
        """
            Return a simplified copy of the environment where distant vehicles have been removed from the road.

            This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, self.PERCEPTION_DISTANCE)

        return state_copy

    def change_vehicles(self, vehicle_class_path):
        """
            Change the type of all vehicles on the road
        :param vehicle_class_path: The path of the class of behavior for other vehicles
                             Example: "highway_env.vehicle.behavior.IDMVehicle"
        :return: a new environment with modified behavior model for other vehicles
        """
        vehicle_class = utils.class_from_path(vehicle_class_path)

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle and not isinstance(v, Obstacle):
                vehicles[i] = vehicle_class.create_from(v)
        return env_copy

    def set_preferred_lane(self, preferred_lane=None):
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    v.route = [(lane[0], lane[1], preferred_lane) for lane in v.route]
                    # Vehicle with lane preference are also less cautious
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
        return env_copy

    def set_route_at_intersection(self, _to):
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.set_route_at_intersection(_to)
        return env_copy

    def randomize_behaviour(self):
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()
        return env_copy

    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1/self.config["policy_frequency"])

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
