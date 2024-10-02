from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle


class RacetrackEnv(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road"],
                    "grid_size": [[-18, 18], [-18, 18]],
                    "grid_step": [3, 3],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 1,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
                "speed_limit": 10.0,
            }
        )
        return config

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> dict[str, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return {
            "lane_centering_reward": 1
            / (1 + self.config["lane_centering_cost"] * lateral**2),
            "action_reward": np.linalg.norm(action),
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane(
            [42, 0],
            [100, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimits[1],
        )
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, 5],
                [100, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[2],
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + 5,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # 3 - Vertical Straight
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [120, -20],
                [120, -30],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [125, -20],
                [125, -30],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[4],
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2 + 5,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[4],
            ),
        )

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3 + 5,
                np.deg2rad(0),
                np.deg2rad(136),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[5],
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3,
                np.deg2rad(0),
                np.deg2rad(137),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[5],
            ),
        )

        # 6 - Slant
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [55.7, -15.7],
                [35.7, -35.7],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [59.3934, -19.2],
                [39.3934, -39.2],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(315),
                np.deg2rad(170),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(315),
                np.deg2rad(165),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(170),
                np.deg2rad(56),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(170),
                np.deg2rad(58),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(240),
                np.deg2rad(270),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[8],
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(238),
                np.deg2rad(268),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[8],
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", rng.integers(2))
                if i == 0
                else self.road.network.random_lane_index(rng)
            )
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road, lane_index, speed=None, longitudinal=rng.uniform(20, 50)
            )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        if self.config["other_vehicles"] > 0:
            # Front vehicle
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                ("b", "c", lane_index[-1]),
                longitudinal=rng.uniform(
                    low=0, high=self.road.network.get_lane(("b", "c", 0)).length
                ),
                speed=6 + rng.uniform(high=3),
            )
            self.road.vehicles.append(vehicle)

            # Other vehicles
            for i in range(rng.integers(self.config["other_vehicles"])):
                random_lane_index = self.road.network.random_lane_index(rng)
                vehicle = IDMVehicle.make_on_lane(
                    self.road,
                    random_lane_index,
                    longitudinal=rng.uniform(
                        low=0, high=self.road.network.get_lane(random_lane_index).length
                    ),
                    speed=6 + rng.uniform(high=3),
                )
                # Prevent early collisions
                for v in self.road.vehicles:
                    if np.linalg.norm(vehicle.position - v.position) < 20:
                        break
                else:
                    self.road.vehicles.append(vehicle)


class RacetrackEnvLarge(RacetrackEnv):
    """
    A larger racetrack map, with three lanes

    credit: @m-walters
    """

    def _make_road(self) -> None:
        net = RoadNetwork()
        w = 5
        w2 = 2 * w
        default_speedlimit = self.config["speed_limit"]

        # Initialise First Lane
        lane = StraightLane(
            [42, 0],
            [200, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=w,
            speed_limit=default_speedlimit,
        )
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, w],
                [200, w],
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, w2],
                [200, w2],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 2 - Circular Arc #1
        center1 = [200, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + w,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + w2,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 3 - Vertical Straight
        delta_extension = -1.0  # Better join
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [220, -20],
                [220, -60 + delta_extension],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [220 + w, -20],
                [220 + w, -60 + delta_extension],
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [220 + w2, -20],
                [220 + w2, -60 + delta_extension],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 4 - Circular Arc #2
        center4 = [205, -60]
        radii4 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center4,
                radii4 + w,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center4,
                radii4 + w2,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 5 - Circular Arc #3
        center5 = [170, -60]
        radii5 = 15
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(0),
                np.deg2rad(136),
                width=w,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(0),
                np.deg2rad(137),
                width=w,
                clockwise=True,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center5,
                radii5 - w,
                np.deg2rad(0),
                np.deg2rad(137),
                width=w,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 6 - Slant
        # Extending [-30,-30]
        extend = np.array([-30, -30])
        start6 = np.array([155.7, -45.7])
        end6 = np.array([135.7, -65.7]) + extend
        net.add_lane(
            "f",
            "g",
            StraightLane(
                start6,
                end6,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        start6_2 = np.array([159.3934, -49.2])
        end6_2 = np.array([139.3934, -69.2]) + extend
        net.add_lane(
            "f",
            "g",
            StraightLane(
                start6_2,
                end6_2,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        start6_3 = np.array(
            [
                start6[0] + 2 * (start6_2[0] - start6[0]),
                start6[1] + 2 * (start6_2[1] - start6[1]),
            ]
        )
        end6_3 = np.array(
            [
                end6[0] + 2 * (end6_2[0] - end6[0]),
                end6[1] + 2 * (end6_2[1] - end6[1]),
            ]
        )
        net.add_lane(
            "f",
            "g",
            StraightLane(
                start6_3,
                end6_3,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 7 - Circular Arc #4
        # Reflect it with the slant
        center7 = np.array([118.1, -48.1]) + extend
        radii7 = 25
        theta7 = 317
        # theta7_end = 270 - (theta7 - 270) - 10
        theta7_end = 205
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center7,
                radii7,
                np.deg2rad(theta7),
                np.deg2rad(theta7_end - 3),  # nicer
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center7,
                radii7 + 5,
                np.deg2rad(theta7),
                np.deg2rad(theta7_end),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center7,
                radii7 + w2,
                np.deg2rad(theta7),
                np.deg2rad(theta7_end),
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 8 - Next slant
        # Reflected from the last arc's center
        start8 = np.array(
            [
                center7[0] + radii7 * np.cos(np.deg2rad(theta7_end)),
                center7[1] + radii7 * np.sin(np.deg2rad(theta7_end)),
            ]
        )
        start8_2 = np.array(
            [
                center7[0] + (radii7 + w) * np.cos(np.deg2rad(theta7_end)),
                center7[1] + (radii7 + w) * np.sin(np.deg2rad(theta7_end)),
            ]
        )
        start8_3 = np.array(
            [
                center7[0] + (radii7 + w2) * np.cos(np.deg2rad(theta7_end)),
                center7[1] + (radii7 + w2) * np.sin(np.deg2rad(theta7_end)),
            ]
        )

        # We preemptively take section 9's radius to make a nice join.
        radii9 = 15
        rad = np.deg2rad(30)
        end8 = np.array([42 - radii9 * np.cos(rad), -radii9 - radii9 * np.sin(rad)])
        end8_2 = np.array(
            [42 - (radii9 + w) * np.cos(rad), -radii9 - (radii9 + w) * np.sin(rad)]
        )
        end8_3 = np.array(
            [42 - (radii9 + w2) * np.cos(rad), -radii9 - (radii9 + w2) * np.sin(rad)]
        )
        net.add_lane(
            "h",
            "i",
            StraightLane(
                start8,
                end8,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "h",
            "i",
            StraightLane(
                start8_2,
                end8_2,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "h",
            "i",
            StraightLane(
                start8_3,
                end8_3,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 9 - Circular arc 7, end
        # Since y2 = 0...
        center9 = np.array([42, -radii9])
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center9,
                radii9,
                np.deg2rad(210),
                np.deg2rad(88),
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center9,
                radii9 + w,
                np.deg2rad(210),
                np.deg2rad(90),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center9,
                radii9 + w2,
                np.deg2rad(212),
                np.deg2rad(88),  # nicer join
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
