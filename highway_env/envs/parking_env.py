from abc import abstractmethod
from typing import Optional

from gymnasium import Env
import numpy as np
import random

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation, observation_factory
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle
from highway_env.utils import are_polygons_intersecting


class GoalEnv(Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class ParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    # For parking env with GrayscaleObservation, the env need
    # this PARKING_OBS to calculate the reward and the info.
    # Bug fixed by Mcfly(https://github.com/McflyWZX)
    PARKING_OBS = {"observation": {
            "type": "KinematicsGoal",
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False
        }}

    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        super().__init__(config, render_mode)
        self.observation_type_parking = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -500,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1,
            "vehicles_count": 0,
            "add_walls": True,
            "center_window": False,
            "n_rows": 2, # number of parking rows
            "y_offset": 10, # y distance between parallel rows
            "font_size": 22, # display font size for ids ['None' to stop render]
            "spots": 14, # spots in each row
            "obstacles": False,
            "random_start": False, 
            "prevent_early_collision": False
        })
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(self, self.PARKING_OBS["observation"])

    def _info(self, obs, action) -> dict:
        info = super(ParkingEnv, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self, spots: int = 14) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        spots = self.config["spots"]
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = self.config["y_offset"]
        length = 8

        row_ys = []
        for i in range(self.config["n_rows"]):
            if i == 0:
                row_ys.append([y_offset, y_offset+length])
            elif i == 1:
                row_ys.append([-y_offset, -y_offset-length])
            else:
                if i%2 == 0:
                    row_ys.append([row_ys[i-2][1]+2*y_offset, row_ys[i-2][1]+2*y_offset+length])
                else:
                    row_ys.append([row_ys[i-2][1]-2*y_offset, row_ys[i-2][1]-2*y_offset-length])

        id = 0
        for row in range(0, self.config["n_rows"]):
            for k in range(spots):
                x = (k + 1 - spots // 2) * (width + x_offset) - width / 2
                net.add_lane(
                    chr(ord('a')+row), chr(ord('a')+row+1), 
                    StraightLane([x, row_ys[row][0]], [x, row_ys[row][1]], 
                                 width=width, line_types=lt, identifier=id, 
                                 display_font_size=self.config['font_size']
                    )
                )
                id += 1

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])
        
        # Store the allowed x,y coordinate ranges to spawn the agent
        allowed_y_space = [[-y_offset+8, y_offset-8]]
        if len(row_ys) > 2:
            for i in range(2, len(row_ys)):
                if i%2 == 0:
                    allowed_y_space.append([row_ys[i-2][1]+8, row_ys[i][0]-8])
                else:
                    allowed_y_space.append([row_ys[i][0]+8, row_ys[i-2][1]-8])

        self.allowed_vehicle_space = {
            'x' : (-30, 30),
            'y' : allowed_y_space
        }

        # Walls
        x_end = abs((1 - spots // 2) * (width + x_offset) - width / 2)

        if len(row_ys) > 1:
            wall_y_top = row_ys[-1][1] + 4 if row_ys[-1][1] > 0 else row_ys[-1][1] - 4
            wall_y_bottom = row_ys[-2][1] + 4 if row_ys[-2][1] > 0 else row_ys[-2][1] - 4
        else:
            wall_y_bottom = row_ys[-1][1] + 4
            wall_y_top = -y_offset - 4

        wall_x = x_end + 14

        for y in [wall_y_top, wall_y_bottom]:
            obstacle = Obstacle(self.road, [0, y])
            obstacle.LENGTH, obstacle.WIDTH = (2*wall_x, 1)
            obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)

        wall_y = 0
        if self.config["n_rows"] > 1 and self.config["n_rows"]%2==1: wall_y = y_offset+4
        elif self.config["n_rows"] == 1: wall_y = y_offset-6

        for x in [-wall_x, wall_x]:
            obstacle = Obstacle(
                self.road, 
                [x, wall_y], 
                heading=np.pi / 2
            )
            obstacle.LENGTH, obstacle.WIDTH = (abs(wall_y_top) + abs(wall_y_bottom), 1)
            obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)

        if self.config['obstacles'] and self.config["n_rows"] > 3:
            self._create_obstacles()

    def _create_obstacles(self):
        """Create some random obstacles"""
        obstacle = Obstacle(self.road, (18, -18 - self.config['y_offset']), 90)
        obstacle.LENGTH, obstacle.WIDTH = 5, 5
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        obstacle.line_color = (187, 84, 49)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, (-18, 18 + self.config['y_offset']), 90)
        obstacle.LENGTH, obstacle.WIDTH = 5, 5
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        obstacle.line_color = (187, 84, 49)
        self.road.objects.append(obstacle)

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            if self.config["random_start"]:
                while(True):
                    x = np.random.randint(self.allowed_vehicle_space['x'][0], self.allowed_vehicle_space['x'][1])
                    y_sector = np.random.choice(range(len(self.allowed_vehicle_space['y'])))
                    y = np.random.randint(self.allowed_vehicle_space['y'][y_sector][0], self.allowed_vehicle_space['y'][y_sector][1])
                    vehicle = self.action_type.vehicle_class(self.road, [x, y], 2*np.pi*self.np_random.uniform(), 0)

                    intersect = False
                    for o in self.road.objects:
                        res, _, _ = are_polygons_intersecting(vehicle.polygon(), o.polygon(), vehicle.velocity, o.velocity)
                        intersect |= res
                    if not intersect:
                        break
            else:
                vehicle = self.action_type.vehicle_class(self.road, [0, 0], 2*np.pi*self.np_random.uniform(), 0)

            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)

        # Goal
        goal_lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = Landmark(self.road, goal_lane.position(goal_lane.length/2, 0), heading=goal_lane.heading)
        self.road.objects.append(self.goal)

        # Other vehicles
        free_lanes = self.road.network.lanes_list().copy()
        free_lanes.remove(goal_lane)
        random.Random(4).shuffle(free_lanes)
        for _ in range(self.config["vehicles_count"] - 1):
            lane = free_lanes.pop()
            v = Vehicle.make_on_lane(self.road, lane, 4, speed=0)
            self.road.vehicles.append(v)

        if self.config["prevent_early_collision"]:
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not self.vehicle and (
                        np.linalg.norm(v.position - self.goal.position) < 20 or
                        np.linalg.norm(v.position - self.vehicle.position) < 20):
                    self.road.vehicles.remove(v)


    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {}) for agent_obs in obs)
        reward += self.config['collision_reward'] * sum(v.crashed for v in self.controlled_vehicles)
        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached or time is over."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        return bool(crashed or success)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time is over."""
        return self.time >= self.config["duration"]


class ParkingEnvActionRepeat(ParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


class ParkingEnvParkedVehicles(ParkingEnv):
    def __init__(self):
        super().__init__({"vehicles_count": 10})
