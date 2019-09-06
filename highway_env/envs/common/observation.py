from __future__ import division, print_function, absolute_import
import pandas
from gym import spaces
import numpy as np

from highway_env import utils
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.control import MDPVehicle
from highway_env.road.graphics import WorldSurface


class ObservationType(object):
    def space(self):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()


class GrayscaleObservation(ObservationType):
    """
        An observation class that collects directly what the simulator renders
        as the input, and stacks the collected frames just as in the nature DQN
        . Specific keys are expected in the configuration dictionnary passed.

        Example of observation dictionnary in the environment config:
            observation": {
                "type": "GrayscaleObservation",
                "weights": [0.2989, 0.5870, 0.1140],  #weights for RGB conversion,
                "stack_size": 4,
                "observation_shape": (84, 84)
            }

        Also, the screen_height and screen_width of the environment should match the
        expected observation_shape. 
    """
    def __init__(self, env, **config):
        self.env = env
        self.config = config
        self.observation_shape = config["observation_shape"]
        self.shape = self.observation_shape + (config["stack_size"], )
        self.state = np.zeros(self.shape)

    def space(self):
        try:
            return spaces.Box(shape=self.shape,
                              low=0, high=1,
                              dtype=np.float32)
        except AttributeError:
            return None

    def observe(self):
        new_obs = self._record_to_grayscale()
        new_obs = np.reshape(new_obs, self.observation_shape)
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[:, :, -1] = new_obs
        return self.state

    def _record_to_grayscale(self):
        raw_rgb = self.env.render('rgb_array')
        return np.dot(raw_rgb[..., :3],
                      self.config['weights'])


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env, horizon=10, **kwargs):
        self.env = env
        self.horizon = horizon

    def space(self):
        try:
            return spaces.Box(shape=self.observe().shape, low=0, high=1, dtype=np.float32)
        except AttributeError:
            return None

    def observe(self):
        grid = compute_ttc_grid(self.env, time_quantization=1/self.env.config["policy_frequency"], horizon=self.horizon)
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.env.vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.env.vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0:lf+1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_velocities = 3
        v0 = grid.shape[0] + self.env.vehicle.velocity_index - obs_velocities // 2
        vf = grid.shape[0] + self.env.vehicle.velocity_index + obs_velocities // 2
        clamped_grid = padded_grid[v0:vf + 1, :, :]
        return clamped_grid


class KinematicObservation(ObservationType):
    """
        Observe the kinematics of nearby vehicles.
    """
    FEATURES = ['presence', 'x', 'y', 'vx', 'vy']

    def __init__(self, env,
                 features=FEATURES,
                 vehicles_count=5,
                 features_range=None,
                 absolute=False,
                 flatten=False,
                 observe_intentions=False,
                 **kwargs):
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param flatten: Flatten the observation to a vector
        :param observe_intentions: Observe the destinations of other vehicles
        """
        self.env = env
        self.features = features
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.flatten = flatten
        self.observe_intentions = observe_intentions

    def space(self):
        shape = (self.vehicles_count * len(self.features),) if self.flatten \
            else (self.vehicles_count, len(self.features))
        return spaces.Box(shape=shape, low=-1, high=1, dtype=np.float32)

    def normalize(self, df):
        """
            Normalize the observation values.

            For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.env.vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * MDPVehicle.SPEED_MAX, 5.0 * MDPVehicle.SPEED_MAX],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
                "vy": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.remap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self):
        # Add ego-vehicle
        df = pandas.DataFrame.from_records([self.env.vehicle.to_dict()])[self.features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.env.vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         self.vehicles_count - 1)
        if close_vehicles:
            origin = self.env.vehicle if not self.absolute else None
            df = df.append(pandas.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize
        df = self.normalize(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pandas.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        # Clip
        obs = np.clip(df.values, -1, 1)
        self.env.np_random.shuffle(obs[1:])
        # Flatten
        if self.flatten:
            obs = np.ravel(obs)
        return obs


class OccupancyGridObservation(ObservationType):
    """
        Observe an occupancy grid of nearby vehicles.
    """
    FEATURES = ['presence', 'vx', 'vy']
    GRID_SIZE = [[-5.5*5, 5.5*5], [-5.5*5, 5.5*5]]
    GRID_STEP = [5, 5]

    def __init__(self,
                 env,
                 features=FEATURES,
                 grid_size=GRID_SIZE,
                 grid_step=GRID_STEP,
                 features_range=None,
                 absolute=False,
                 **kwargs):
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        """
        self.env = env
        self.features = features
        self.grid_size = np.array(grid_size)
        self.grid_step = np.array(grid_step)
        grid_shape = np.asarray(np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / grid_step), dtype=np.int)
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.features_range.pop("x", None)
        self.features_range.pop("y", None)
        self.absolute = absolute

    def space(self):
        return spaces.Box(shape=self.grid.shape, low=-1, high=1, dtype=np.float32)

    def normalize(self, df):
        """
            Normalize the observation values.

            For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
                "vy": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.remap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self):
        if self.absolute:
            raise NotImplementedError()
        else:
            # Add nearby traffic
            self.grid.fill(0)
            df = pandas.DataFrame.from_records(
                [v.to_dict(self.env.vehicle) for v in self.env.road.vehicles if v is not self.env.vehicle])
            # Normalize
            df = self.normalize(df)
            # Fill-in features
            for layer, feature in enumerate(self.features):
                for _, vehicle in df.iterrows():
                    cell = (int((vehicle["x"] - self.grid_size[0, 0]) / self.grid_step[0]),
                            int((vehicle["y"] - self.grid_size[1, 0]) / self.grid_step[1]))
                    if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                        self.grid[layer, cell[1], cell[0]] = vehicle[feature]
            # Stack ego-velocity
            # obs = np.stack(np.ravel(self.grid),
            #                np.array(self.env.vehicle.velocity / MDPVehicle.SPEED_MAX))

            # Clip
            obs = np.clip(self.grid, -1, 1)
            return obs


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env, scales, **kwargs):
        self.scales = np.array(scales)
        super(KinematicsGoalObservation, self).__init__(env, **kwargs)

    def space(self):
        try:
            obs = self.observe()
            return spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
            ))
        except AttributeError:
            return None

    def observe(self):
        obs = np.ravel(pandas.DataFrame.from_records([self.env.vehicle.to_dict()])[self.features])
        goal = np.ravel(pandas.DataFrame.from_records([self.env.goal.to_dict()])[self.features])
        obs = {
            "observation": obs / self.scales,
            "achieved_goal": obs / self.scales,
            "desired_goal": goal / self.scales
        }
        return obs


def observation_factory(env, config):
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
