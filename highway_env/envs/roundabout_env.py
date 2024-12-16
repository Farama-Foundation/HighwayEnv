from __future__ import annotations

from random import shuffle

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lanes.unweighted_lanes import CircularLane, SineLane, StraightLane
from highway_env.road.lanes.lane_utils import LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle

class RoundaboutEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "absolute": True,
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-15, 15],
                        "vy": [-15, 15],
                    },
                },
                "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 8, 16]},
                "reward_speed_range": [8, 16],
                "incoming_vehicle_destination": None,
                "collision_reward": -10,
                "high_speed_reward": 0.4,
                "right_lane_reward": 0,
                "lane_change_reward": -0.05,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "duration": 30,
                "normalize_reward": True,
                "vehicles_count": 10,
                # Reward weights
                "collision_weight": 1,
                "distance_from_goal_weight": 1,
                "lane_change_weight": 1,
                "headway_evaluation_weight": 1,
                "on_road_weight": 1,
            }
        )
        return config

    def _reward(self, action: int) -> float:
        MIN_REWARD = -10
        MAX_REWARD = 5
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [MIN_REWARD, MAX_REWARD],
                [0, 1],
            )

        return reward

    def _rewards(self, action: int) -> dict[str, float]:
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": self.vehicle.crashed * self.config["collision_reward"],
            "distance_from_goal": self.vehicle.remaining_route_nodes,
            "lane_change_reward": float(action in [0, 2]),
            "headway_evaluation": self.vehicle.headway_evaluation,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"] or self.vehicle.remaining_route_nodes == 1

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius, radius + 4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane(
                "se",
                "ex",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 - alpha),
                    np.deg2rad(alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ex",
                "ee",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(alpha),
                    np.deg2rad(-alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ee",
                "nx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-alpha),
                    np.deg2rad(-90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "nx",
                "ne",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-90 + alpha),
                    np.deg2rad(-90 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ne",
                "wx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-90 - alpha),
                    np.deg2rad(-180 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "wx",
                "we",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-180 + alpha),
                    np.deg2rad(-180 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "we",
                "sx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(180 - alpha),
                    np.deg2rad(90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "sx",
                "se",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 + alpha),
                    np.deg2rad(90 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2 * dev  # [m]

        delta_en = dev - delta_st
        w = 2 * np.pi / dev
        net.add_lane(
            "ser", "ses", StraightLane([2, access], [2, dev / 2], line_types=(s, c))
        )
        net.add_lane(
            "ses",
            "se",
            SineLane(
                [2 + a, dev / 2],
                [2 + a, dev / 2 - delta_st],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "sx",
            "sxs",
            SineLane(
                [-2 - a, -dev / 2 + delta_en],
                [-2 - a, dev / 2],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c))
        )

        net.add_lane(
            "sxr", "sxre", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c))
        )

        net.add_lane(
            "eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c))
        )
        net.add_lane(
            "ees",
            "ee",
            SineLane(
                [dev / 2, -2 - a],
                [dev / 2 - delta_st, -2 - a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "ex",
            "exs",
            SineLane(
                [-dev / 2 + delta_en, 2 + a],
                [dev / 2, 2 + a],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c))
        )

        net.add_lane(
            "exr", "exre", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c))
        )

        net.add_lane(
            "ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c))
        )
        net.add_lane(
            "nes",
            "ne",
            SineLane(
                [-2 - a, -dev / 2],
                [-2 - a, -dev / 2 + delta_st],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "nx",
            "nxs",
            SineLane(
                [2 + a, dev / 2 - delta_en],
                [2 + a, -dev / 2],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c))
        )

        net.add_lane(
            "nxr", "nxre", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c))
        )

        net.add_lane(
            "wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c))
        )
        net.add_lane(
            "wes",
            "we",
            SineLane(
                [-dev / 2, 2 + a],
                [-dev / 2 + delta_st, 2 + a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "wx",
            "wxs",
            SineLane(
                [dev / 2 - delta_en, -2 - a],
                [-dev / 2, -2 - a],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c))
        )

        net.add_lane(
            "wxr", "wxre", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c))
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _get_random_entry(self) -> tuple[str, str, int]:
        """
        Gives a random entry as a triple (u, v, lane_idx), where (u, v) is an edge.
        """
        entries = self._get_entry_edges()
        entry = self.np_random.choice(entries)
        return tuple((entry[0], entry[1], 0))

    def _get_random_edge_from(self, edge_list: list[tuple[str, str, int]]) -> tuple[str, str, int]:
        edge = self.np_random.choice(edge_list)
        return tuple((edge[0], edge[1], 0))

    def _get_entry_edges(self) -> list[tuple[str, str, int]]:
        """
        Returns the entry edges of the roundabout.
        """
        return [
            ("ner", "nes", 0),
            ("ser", "ses", 0),
            ("eer", "ees", 0),
            ("wer", "wes", 0),
        ]


    def _get_random_exit(self) -> tuple[str, str, int]:
        """
        Gives a random exit as a triple (u, v, lane_idx), where (u, v) is an edge.
        """
        exits = np.array([("nxs", "nxr"), ("sxs", "sxr"), ("exs", "exr"), ("wxs", "wxr")])
        road_exit = self.np_random.choice(exits)
        return tuple((road_exit[0], road_exit[1], 0))

    def _get_random_roundabout_lane_index(self):
        lane_indices = np.array([("sx", "se"), ("we", "sx"), ("wx", "we"), ("ne", "wx"), ("nx", "ne"), ("ee", "nx"), ("ex", "ee"), ("se", "ex")])
        lane_index = self.np_random.choice(lane_indices)
        return tuple((lane_index[0], lane_index[1], self.np_random.integers(0,2)))

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        position_deviation = 2
        speed_deviation = 2

        # Ego-vehicle
        ego_lane = self.road.network.get_lane(self._get_random_entry())
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(125, 0),
            speed=8,
            heading=ego_lane.heading_at(140),
        )
        try:
            ego_vehicle.plan_route_to(self._get_random_exit()[1])
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Other vehicles
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # Spawning 2 vehicles in the roundabout
        for i in [1, -1]:
            vehicle = other_vehicles_type.make_on_lane(
                self.road,
                self._get_random_roundabout_lane_index(),
                longitudinal=20 * i + self.np_random.normal() * position_deviation,
                speed=16 + self.np_random.normal() * speed_deviation,
            )
            vehicle.plan_route_to(self._get_random_exit()[1])
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Entering vehicles
        spawns: dict[tuple[str, str, int], int] = {
            ('ner', 'nes', 0): 0,
            ('ser', 'ses', 0): 0,
            ('eer', 'ees', 0): 0,
            ('wer', 'wes', 0): 0,
        }
        entries = self._get_entry_edges()
        max_vehicles_pr_lane = 8
        if self.config['vehicles_count'] > max_vehicles_pr_lane * 4:
            self.config['vehicles_count'] = max_vehicles_pr_lane * 4
        for i in range(self.config["vehicles_count"]):
            # Ensuring that no more than 8 vehicles are spawned pr. lane
            entry = self._get_random_edge_from(entries)
            while spawns[entry] > max_vehicles_pr_lane:
                entries.remove(entry)
                entry = self._get_random_edge_from(entries)

            spawns[entry] += 1
            longitudinal_deviation = spawns[entry] % max_vehicles_pr_lane
            exit_edge = self._get_random_exit()
            vehicle = other_vehicles_type.make_on_lane(
                self.road,
                entry,
                longitudinal=15 * longitudinal_deviation + self.np_random.normal() * position_deviation,
                speed=16 + self.np_random.normal() * speed_deviation,
            )
            vehicle.plan_route_to(exit_edge[1])
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)
