from __future__ import division, print_function, absolute_import
from gym.envs.registration import register
from gym import GoalEnv, spaces
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.dynamics import Vehicle, Obstacle


class SummonEnv(AbstractEnv, GoalEnv):
    """
        A continuous control environment.

        It implements a reach-type task, where the agent observes their position and velocity and must
        control their acceleration and steering so as to reach a given goal.

        Credits to Vinny Ruia for the idea and initial implementation.
    """
    
    COLLISION_REWARD = -5

    STEERING_RANGE = np.pi / 4
    ACCELERATION_RANGE = 5.0

    REWARD_WEIGHTS = np.array([1, 0.3, 0, 0, 0.02, 0.02])
    SUCCESS_GOAL_REWARD = 0.12

    def define_spaces(self):
        super().define_spaces()
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "vehicles_count": 10,
            "policy_frequency": 5,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5]
        })
        return config

    def step(self, action):
        # Forward action to the vehicle

        self.vehicle.act({
            "acceleration": action[0] * self.ACCELERATION_RANGE,
            "steering": action[1] * self.STEERING_RANGE
        })
        self._simulate()

        obs = self.observation.observe()
        info = {"is_success": self._is_success(obs['achieved_goal'], obs['desired_goal'])}
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        terminal = self._is_terminal(obs)
        return obs, reward, terminal, info

    def reset(self):
        self._create_road()
        self._create_vehicles()
        return super(SummonEnv, self).reset()

    def _create_road(self, spots=15):
        """
            Create a road composed of straight adjacent lanes.
        """
        net = RoadNetwork()

        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 12
        length = 8
        # Parking spots
        for k in range(spots):
            x = (k - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset + length],
                                                width=width, line_types=lt, speed_limit=5))
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset - length],
                                                width=width, line_types=lt, speed_limit=5))

        self.spots = spots
        self.vehicle_starting = [x, y_offset + (length / 2)]
        self.num_middle_lanes = 0
        self.x_range = (int(spots / 2) + 1) * width

        # Generate the middle lane for the busy parking lot
        for y in np.arange(-y_offset + width, y_offset, width):
            net.add_lane("d", "e", StraightLane([-self.x_range, y], [self.x_range, y],
                                                width=width,
                                                line_types=(LineType.STRIPED, LineType.STRIPED),
                                                speed_limit=5))
            self.num_middle_lanes += 1

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _create_vehicles(self, parked_probability=0.75):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """

        self.vehicle = Vehicle(self.road, self.vehicle_starting, 2 * np.pi * self.np_random.rand(), 0)
        self.road.vehicles.append(self.vehicle)

        goal_position = [self.np_random.choice([-2 * self.spots - 10, 2 * self.spots + 10]), 0]
        self.goal = Obstacle(self.road, goal_position, heading=0)
        self.goal.COLLISIONS_ENABLED = False
        self.road.vehicles.insert(0, self.goal)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(self.config["vehicles_count"]):
            is_parked = self.np_random.rand() <= parked_probability
            if not is_parked:
                # Just an effort to spread the vehicles out
                idx = self.np_random.randint(0, self.num_middle_lanes)
                longitudinal = (i * 5) - (self.x_range / 8) * self.np_random.randint(-1, 1)
                self.road.vehicles.append(
                    vehicles_type.make_on_lane(self.road, ("d", "e", idx), longitudinal, velocity=2))
            else:
                lane = ("a", "b", i) if self.np_random.rand() >= 0.5 else ("b", "c", i)
                self.road.vehicles.append(Vehicle.make_on_lane(self.road, lane, 4, velocity=0))

        for v in self.road.vehicles:  # Prevent early collisions
            if v is not self.vehicle and np.linalg.norm(v.position - self.vehicle.position) < 20:
                self.road.vehicles.remove(v)

    def compute_reward(self, achieved_goal, desired_goal, info, p=0.5):
        """
            Proximity to the goal is rewarded

            We use a weighted p-norm
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return - np.power(np.dot(np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p) + (self.COLLISION_REWARD * self.vehicle.crashed)

    def _reward(self, action):
        raise Exception("Use compute_reward instead, as for GoalEnvs")

    def _is_success(self, achieved_goal, desired_goal):
        return self.compute_reward(achieved_goal, desired_goal, None) > -self.SUCCESS_GOAL_REWARD

    def _is_terminal(self, obs=None):
        """
            The episode is over if the ego vehicle crashed or the goal is reached.
        """
        done = self.vehicle.crashed
        if obs is not None:
            done = done or self._is_success(obs['achieved_goal'], obs['desired_goal'])
        return done


class SummonEnvActionRepeat(SummonEnv):
    def __init__(self):
        super().__init__()
        self.configure({"policy_frequency": 1})


register(
    id='summon-v0',
    entry_point='highway_env.envs:SummonEnv',
    max_episode_steps=100
)

register(
    id='summon-ActionRepeat-v0',
    entry_point='highway_env.envs:SummonEnvActionRepeat',
    max_episode_steps=20
)
