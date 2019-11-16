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

        Credits to Munir Jojo-Verge for the idea and initial implementation.
    """
    
    COLLISION_REWARD = -1
    HIGH_VELOCITY_REWARD = 0.2
    RIGHT_LANE_REWARD = 0
    LANE_CHANGE_REWARD = -0.05

    STEERING_RANGE = np.pi / 4
    ACCELERATION_RANGE = 5.0

    REWARD_WEIGHTS = np.array([1, 0.3, 0, 0, 0.02, 0.02])
    SUCCESS_GOAL_REWARD = 0.12

    def __init__(self):
        super().__init__()
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
            "incoming_vehicle_destination": None,
            "policy_frequency": 5,
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
        y_offset = 10
        length = 8
        for k in range(spots):
            x = (k - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _create_vehicles(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        position_deviation = 2
        velocity_deviation = 2

        self.vehicle = Vehicle(self.road, [0, 0], 2*np.pi*self.np_random.rand(), 0)
        self.road.vehicles.append(self.vehicle)

        lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = Obstacle(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
        self.goal.COLLISIONS_ENABLED = False
        self.road.vehicles.insert(0, self.goal)

         # Incoming vehicle
        destinations = ["exr", "sxr", "nxr"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("a", "b", 1),
                                                   longitudinal=5 + self.np_random.randn()*position_deviation,
                                                   velocity=16 + self.np_random.randn()*velocity_deviation)

        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in list(range(1, 2)) + list(range(-1, 0)):
            vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("a", "b", 0),
                                                       longitudinal=20*i + self.np_random.randn()*position_deviation,
                                                       velocity=16 + self.np_random.randn()*velocity_deviation)
            vehicle.plan_route_to(self.np_random.choice(destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Entering vehicle
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("b", "c", 0),
                                                   longitudinal=50 + self.np_random.randn() * position_deviation,
                                                   velocity=16 + self.np_random.randn() * velocity_deviation)
        vehicle.plan_route_to(self.np_random.choice(destinations))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)


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
        return - np.power(np.dot(np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p)

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
        super().__init__({"policy_frequency": 1})


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
