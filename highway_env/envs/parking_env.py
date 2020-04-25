from gym.envs.registration import register
from gym import GoalEnv
import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle, Obstacle


class ParkingEnv(AbstractEnv, GoalEnv):
    """
        A continuous control environment.

        It implements a reach-type task, where the agent observes their position and velocity and must
        control their acceleration and steering so as to reach a given goal.

        Credits to Munir Jojo-Verge for the idea and initial implementation.
    """
    REWARD_WEIGHTS = np.array([1, 0.3, 0, 0, 0.02, 0.02])
    SUCCESS_GOAL_REWARD = 0.12

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
            "action": {
                "type": "Continuous"
            },
            "policy_frequency": 5,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5]
        })
        return config

    def step(self, action):
        obs, reward, terminal, info = super().step(action)
        info.update({"is_success": self._is_success(obs['achieved_goal'], obs['desired_goal'])})
        return obs, reward, terminal, info

    def reset(self):
        self._create_road()
        self._create_vehicles()
        return super().reset()

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
        self.vehicle = Vehicle(self.road, [0, 0], 2*np.pi*self.np_random.rand(), 0)
        self.road.vehicles.append(self.vehicle)

        lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = Obstacle(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
        self.goal.COLLISIONS_ENABLED = False
        self.road.obstacles.append(self.goal)

    def compute_reward(self, achieved_goal, desired_goal, info, p=0.5):
        """
            Proximity to the goal is rewarded

            We use a weighted p-norm
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p)

    def _reward(self, action):
        obs = self.observation.observe()
        return self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})

    def _is_success(self, achieved_goal, desired_goal):
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.SUCCESS_GOAL_REWARD

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the goal is reached.
        """
        obs = self.observation.observe()
        return self.vehicle.crashed or self._is_success(obs['achieved_goal'], obs['desired_goal'])


class ParkingEnvActionRepeat(ParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1})


register(
    id='parking-v0',
    entry_point='highway_env.envs:ParkingEnv',
    max_episode_steps=100
)

register(
    id='parking-ActionRepeat-v0',
    entry_point='highway_env.envs:ParkingEnvActionRepeat',
    max_episode_steps=20
)
