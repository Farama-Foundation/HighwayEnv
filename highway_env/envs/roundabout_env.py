from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, SineLane, StraightLane
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
                "incoming_vehicle_destination": None,
                "collision_reward": -1,
                "high_speed_reward": 0.2,
                "right_lane_reward": 0,
                "lane_change_reward": -0.05,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "duration": 11,
                "normalize_reward": True,
            }
        )
        return config

    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["high_speed_reward"]],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: int) -> dict[str, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward": MDPVehicle.get_speed_index(self.vehicle)
            / (MDPVehicle.DEFAULT_TARGET_SPEEDS.size - 1),
            "lane_change_reward": action in [0, 2],
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

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        position_deviation = 2.0
        speed_deviation = 2.0

        # Ego-vehicle
        ego_lane = self.road.network.get_lane(("ser", "ses", 0))
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(125.0, 0.0),
            speed=8.0,
            heading=ego_lane.heading_at(140.0),
        )
        try:
            ego_vehicle.plan_route_to("nxs")
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Incoming vehicle
        destinations = ["exr", "sxr", "nxr"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.make_on_lane(
            self.road,
            ("we", "sx", 1),
            longitudinal=5.0 + self.np_random.normal() * position_deviation,
            speed=16 + self.np_random.normal() * speed_deviation,
        )

        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in list(range(1, 2)) + list(range(-1, 0)):
            vehicle = other_vehicles_type.make_on_lane(
                self.road,
                ("we", "sx", 0),
                longitudinal=20.0 * float(i)
                + self.np_random.normal() * position_deviation,
                speed=16.0 + self.np_random.normal() * speed_deviation,
            )
            vehicle.plan_route_to(self.np_random.choice(destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Entering vehicle
        vehicle = other_vehicles_type.make_on_lane(
            self.road,
            ("eer", "ees", 0),
            longitudinal=50.0 + self.np_random.normal() * position_deviation,
            speed=16.0 + self.np_random.normal() * speed_deviation,
        )
        vehicle.plan_route_to(self.np_random.choice(destinations))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)


class RoundaboutGenericEnv(RoundaboutEnv):
    """
    A generic version of the roundabout environment.
    Additionally supports changing:
    - the number of lanes in the roundabout
    - the roundabout radius
    - the number of spawned other vehicles
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "roundabout_radius": 20,
                "roundabout_lanes": 2,
                "vehicles_count": 5,
                "duration": 17,
            }
        )
        return config

    def _make_road(self) -> None:
        center = [0, 0]  # [m]
        radius = self.config["roundabout_radius"]
        num_lanes = self.config["roundabout_lanes"]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius + (4 * i) for i in range(num_lanes)]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        nodes = ["se", "ex", "ee", "nx", "ne", "wx", "we", "sx", "se"]

        angles = [
            (90 - alpha, alpha),
            (alpha, -alpha),
            (-alpha, -90 + alpha),
            (-90 + alpha, -90 - alpha),
            (-90 - alpha, -180 + alpha),
            (-180 + alpha, -180 - alpha),
            (180 - alpha, 90 + alpha),
            (90 + alpha, 90 - alpha),
        ]

        for lane in range(num_lanes):
            if num_lanes == 1:
                line_types = [c, c]
            elif lane == 0:
                line_types = [c, s]
            elif lane == num_lanes - 1:
                line_types = [n, c]
            else:
                line_types = [n, s]

            for i in range(8):
                net.add_lane(
                    nodes[i],
                    nodes[i + 1],
                    CircularLane(
                        center,
                        radii[lane],
                        np.deg2rad(angles[i][0]),
                        np.deg2rad(angles[i][1]),
                        clockwise=False,
                        line_types=line_types,
                    ),
                )

        # Dynamically calculate exact coordinates on the outermost circle
        outer_radius = radii[-1]

        def pt(angle_deg: float) -> list[float]:
            rad = np.deg2rad(angle_deg)
            return [outer_radius * np.cos(rad), outer_radius * np.sin(rad)]

        p_se = pt(90 - alpha)
        p_ex = pt(alpha)
        p_ee = pt(-alpha)
        p_nx = pt(-90 + alpha)
        p_ne = pt(-90 - alpha)
        p_wx = pt(-180 + alpha)
        p_we = pt(180 - alpha)
        p_sx = pt(90 + alpha)

        # In case radius is very large
        dev = max(100.0, 2 * outer_radius + 40.0)
        access = dev + 40.0

        # South Entry (ses -> se)
        dy = dev / 2 - p_se[1]
        a = (p_se[0] - 2) / 2
        w = np.pi / dy
        net.add_lane(
            "ser",
            "ses",
            StraightLane([2, access], [2, dev / 2], line_types=(s, c)),
        )
        net.add_lane(
            "ses",
            "se",
            SineLane(
                [2 + a, dev / 2], [2 + a, p_se[1]], a, w, -np.pi / 2, line_types=(c, c)
            ),
        )

        # South Exit (sx -> sxs)
        dy = dev / 2 - p_sx[1]
        a = (p_sx[0] + 2) / 2
        w = np.pi / dy
        net.add_lane(
            "sx",
            "sxs",
            SineLane(
                [p_sx[0] - a, p_sx[1]],
                [p_sx[0] - a, dev / 2],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "sxs",
            "sxr",
            StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)),
        )

        # East Entry (ees -> ee)
        dx = dev / 2 - p_ee[0]
        a = (-2 - p_ee[1]) / 2
        w = np.pi / dx
        net.add_lane(
            "eer",
            "ees",
            StraightLane([access, -2], [dev / 2, -2], line_types=(s, c)),
        )
        net.add_lane(
            "ees",
            "ee",
            SineLane(
                [dev / 2, -2 - a],
                [p_ee[0], -2 - a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )

        # East Exit (ex -> exs)
        dx = dev / 2 - p_ex[0]
        a = (2 - p_ex[1]) / 2
        w = np.pi / dx
        net.add_lane(
            "ex",
            "exs",
            SineLane(
                [p_ex[0], p_ex[1] + a],
                [dev / 2, p_ex[1] + a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "exs",
            "exr",
            StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)),
        )

        # North Entry (nes -> ne)
        dy = p_ne[1] - (-dev / 2)
        a = (-2 - p_ne[0]) / 2
        w = np.pi / dy
        net.add_lane(
            "ner",
            "nes",
            StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)),
        )
        net.add_lane(
            "nes",
            "ne",
            SineLane(
                [-2 - a, -dev / 2],
                [-2 - a, p_ne[1]],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )

        # North Exit (nx -> nxs)
        dy = p_nx[1] - (-dev / 2)
        a = (2 - p_nx[0]) / 2
        w = np.pi / dy
        net.add_lane(
            "nx",
            "nxs",
            SineLane(
                [p_nx[0] + a, p_nx[1]],
                [p_nx[0] + a, -dev / 2],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "nxs",
            "nxr",
            StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)),
        )

        # West Entry (wes -> we)
        dx = p_we[0] - (-dev / 2)
        a = (p_we[1] - 2) / 2
        w = np.pi / dx
        net.add_lane(
            "wer",
            "wes",
            StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)),
        )
        net.add_lane(
            "wes",
            "we",
            SineLane(
                [-dev / 2, 2 + a], [p_we[0], 2 + a], a, w, -np.pi / 2, line_types=(c, c)
            ),
        )

        # West Exit (wx -> wxs)
        dx = p_wx[0] - (-dev / 2)
        a = (p_wx[1] + 2) / 2
        w = np.pi / dx
        net.add_lane(
            "wx",
            "wxs",
            SineLane(
                [p_wx[0], p_wx[1] - a],
                [-dev / 2, p_wx[1] - a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "wxs",
            "wxr",
            StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        speed_deviation = 2.0
        vehicle_count = self.config["vehicles_count"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        destinations = ["exr", "sxr", "nxr", "wxr"]

        ego_lane_id = ("ser", "ses", 0)
        ego_lane = self.road.network.get_lane(ego_lane_id)
        ego_longitudinal = ego_lane.length - 2.5  # Placed at end of straight lane

        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(ego_longitudinal, 0.0),
            speed=8.0,
            heading=ego_lane.heading_at(ego_longitudinal),
        )
        try:
            ego_vehicle.plan_route_to("nxs")
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        spawned_positions = {}
        spawned_positions[ego_lane_id] = [ego_longitudinal]

        spawn_lanes = [
            ("we", "sx"),
            ("sx", "se"),
            ("ee", "nx"),
            ("nx", "ne"),
            ("eer", "ees"),
            ("ner", "nes"),
            ("wer", "wes"),
        ]

        spawned_points = []
        spawned_points.append(ego_lane.position(ego_longitudinal, 0.0))
        safe_distance = 7.0  # safe distance to spawn vehicles from each other
        tries = 10  # number of times it tries to spawn a vehicle
        for _ in range(vehicle_count):
            for _ in range(tries):
                lane_tuple = spawn_lanes[self.np_random.integers(0, len(spawn_lanes))]
                available_lanes = len(
                    self.road.network.graph[lane_tuple[0]][lane_tuple[1]]
                )
                lane_index = self.np_random.integers(0, available_lanes)
                lane_id = (lane_tuple[0], lane_tuple[1], lane_index)

                lane = self.road.network.get_lane(lane_id)
                longitudinal = self.np_random.uniform(
                    5.0,
                    max(5.0, lane.length - 5.0),
                )

                candidate_position = lane.position(longitudinal, 0.0)

                is_safe = True
                for point in spawned_points:
                    if np.linalg.norm(candidate_position - point) < safe_distance:
                        is_safe = False
                        break

                if is_safe:
                    vehicle = other_vehicles_type.make_on_lane(
                        self.road,
                        lane_id,
                        longitudinal=longitudinal,
                        speed=14.0 + self.np_random.normal() * speed_deviation,
                    )

                    if self.config.get("incoming_vehicle_destination") is not None:
                        dest_idx = min(
                            self.config["incoming_vehicle_destination"],
                            len(destinations) - 1,
                        )
                        destination = destinations[dest_idx]
                    else:
                        destination = destinations[
                            self.np_random.integers(0, len(destinations))
                        ]

                    vehicle.plan_route_to(destination)
                    vehicle.randomize_behavior()
                    self.road.vehicles.append(vehicle)

                    spawned_points.append(candidate_position)
                    break
