from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, ConnectedLaneNeighboursMixin
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [20, 30],
                "merging_speed_reward": -0.5,
                "lane_change_reward": -0.05,
            }
        )
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )
        return utils.lmap(
            reward,
            [
                self.config["collision_reward"] + self.config["merging_speed_reward"],
                self.config["high_speed_reward"] + self.config["right_lane_reward"],
            ],
            [0, 1],
        )

    def _rewards(self, action: int) -> dict[str, float]:
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": scaled_speed,
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2)
                and isinstance(vehicle, ControlledVehicle)
            ),
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
                ),
            )

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
            neighbour_vehicles_connected_lanes=self.config[
                "neighbour_vehicles_connected_lanes"
            ],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30.0, 0.0), speed=30.0
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        for position, speed in [(90.0, 29.0), (70.0, 31.0), (5.0, 31.5)]:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + self.np_random.uniform(-5.0, 5.0), 0.0)
            speed += self.np_random.uniform(-1.0, 1.0)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        merging_v = other_vehicles_type(
            road, road.network.get_lane(("j", "k", 0)).position(110.0, 0.0), speed=20.0
        )
        merging_v.target_speed = 30.0
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle


class ConnectedLaneMergeEnv(ConnectedLaneNeighboursMixin, MergeEnv):
    pass


class MergeGenericEnv(MergeEnv):
    """
    A generic version of the merge environment.
    Additionally supports changing:
    - the number of lanes
    - the number of vehicles
    - the size of each section of the merging road

    Visual representation of each configurable merging road segment:
    ======================================
    --------------------------------------
    ======================================
               /  __________/  (after)
              /  / (parallel)
             /  /
    ________/  /(converge)
    __________/
     (before)
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "lanes_count": 2,
                "vehicles_count": 10,
                # Parameters that define the size of each component of the merging road:
                # Section before the merge segment (must be >= 60)
                "before_merge_length": 150,
                # Section converging closer to the highway
                "converge_merge_length": 80,
                # Section where the vehicle can merge into the highway
                "parallel_merge_length": 80,
                # Section after the merge segment (must be >= 100)
                "after_merge_length": 300,
            }
        )
        return cfg

    def _make_road(self) -> None:
        net = RoadNetwork()
        lanes = self.config["lanes_count"]
        pre_merge = self.config["before_merge_length"]
        converge = self.config["converge_merge_length"]
        parallel = self.config["parallel_merge_length"]
        after = self.config["after_merge_length"]

        net = RoadNetwork.straight_road_network(
            lanes,
            start=0,
            length=pre_merge + converge,
            nodes_str=("a", "b"),
            speed_limit=30,
            net=net,
        )
        net = RoadNetwork.straight_road_network(
            lanes,
            start=pre_merge + converge,
            length=parallel,
            nodes_str=("b", "c"),
            speed_limit=30,
            net=net,
        )
        net = RoadNetwork.straight_road_network(
            lanes,
            start=pre_merge + converge + parallel,
            length=after,
            nodes_str=("c", "d"),
            speed_limit=30,
            net=net,
        )

        amplitude = 3.25
        y_parallel = lanes * StraightLane.DEFAULT_WIDTH
        y_approach = y_parallel + 2 * amplitude

        ljk = StraightLane(
            [0, y_approach],
            [pre_merge, y_approach],
            line_types=[LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE],
            forbidden=True,
            speed_limit=30,
        )
        lkb = SineLane(
            [pre_merge, y_parallel + amplitude],
            [pre_merge + converge, y_parallel + amplitude],
            amplitude,
            2 * np.pi / (2 * converge),
            np.pi / 2,
            line_types=[LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE],
            forbidden=True,
            speed_limit=30,
        )
        lbc = StraightLane(
            [pre_merge + converge, y_parallel],
            [pre_merge + converge + parallel, y_parallel],
            line_types=[LineType.STRIPED, LineType.CONTINUOUS_LINE],
            forbidden=True,
            speed_limit=30,
        )

        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", False),
        )
        road.objects.append(Obstacle(road, lbc.position(parallel, 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        road = self.road
        lanes = self.config["lanes_count"]

        pre_merge = self.config["before_merge_length"]
        converge = self.config["converge_merge_length"]
        parallel = self.config["parallel_merge_length"]

        ego_lane = road.network.get_lane(("a", "b", lanes - 1))
        ego_longitudinal = 30.0
        ego_vehicle = self.action_type.vehicle_class(
            road,
            ego_lane.position(ego_longitudinal, 0.0),
            speed=30.0,
        )
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicles_count = self.config["vehicles_count"]
        max_pos = pre_merge + converge + parallel

        spawned_positions = {i: [] for i in range(lanes)}
        spawned_positions[lanes - 1].append(ego_longitudinal)
        safe_distance = 15.0  # safe distance to spawn vehicles from each other
        tries = 10  # number of times it tries to spawn a vehicle
        for _ in range(vehicles_count):
            for _ in range(tries):
                random_lane_index = self.np_random.integers(lanes)
                longitudinal = self.np_random.uniform(0, max_pos)

                if all(
                    abs(longitudinal - p) > safe_distance
                    for p in spawned_positions[random_lane_index]
                ):
                    lane = road.network.get_lane(("a", "b", random_lane_index))
                    pos = lane.position(longitudinal, 0.0)
                    spd = 30.0 + self.np_random.uniform(-2.0, 2.0)

                    road.vehicles.append(other_vehicles_type(road, pos, speed=spd))
                    spawned_positions[random_lane_index].append(longitudinal)
                    break

        merge_lane = road.network.get_lane(("j", "k", 0))
        merging_v = other_vehicles_type(
            road, merge_lane.position(ego_longitudinal + 30, 0.0), speed=20.0
        )
        merging_v.target_speed = 30.0
        road.vehicles.append(merging_v)

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        end_position = (
            self.config["before_merge_length"]
            + self.config["converge_merge_length"]
            + self.config["parallel_merge_length"]
            + self.config["after_merge_length"]
            - 100
        )
        return self.vehicle.crashed or self.vehicle.position[0] > end_position

    def _is_truncated(self) -> bool:
        return False


class ConnectedLaneMergeGenericEnv(ConnectedLaneNeighboursMixin, MergeGenericEnv):
    pass
