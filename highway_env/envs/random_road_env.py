from itertools import chain

import numpy as np

from highway_env.envs.common.abstract import AbstractEnv, Observation
from highway_env.envs.common.action import Action, action_factory
from highway_env.envs.common.observation import observation_factory
from highway_env.road.generation.engine.gen_utils import Lane
from highway_env.road.generation.generator import generate_random_lanes
from highway_env.road.generation.spatial_hash import (
    get_proximal_lanes_wrt_gridpoint,
    point_to_gridpoint,
)
from highway_env.road.lane import PolyLane
from highway_env.road.partitioned_road import PartitionedRoadNetwork
from highway_env.road.road import LineType, Road
from highway_env.vehicle.objects import Landmark, RoadObject


class ParkingSpot(Landmark):
    LENGTH = 7.0
    WIDTH = 3.0


class RandomRoadEnv(AbstractEnv):
    """
    A navigation, negotiation, and parking environment
    set on a procedurally generated road network.

    The goal of an agent is to get to a parking spot
    as soon as possible without crashing into a curb or
    other vehicle.

    Todo:
    - add multi-agent support
    """

    def __init__(
        self,
        config: dict = None,
        render_mode: str | None = None,
        generation_params=None,
        lanes=None,
    ) -> None:
        """
        :param generation_params: Parameter dict for road generation
        :param lanes: Optional list of Lane to load previously generated
        road networks
        """
        self.generation_params = generation_params
        self.lanes = lanes
        super().__init__(config, render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "screen_width": 1200,
                "screen_height": 700,
                "observation": {
                    "type": "TupleObservation",
                    "observation_configs": [
                        {"type": "NavigationObservation"},  # must be first
                        # {"type": "LidarObservation"}, # used for detecting other agents
                        {"type": "LaneLidarObservation"},
                        {"type": "RelativeGoalObservation"},
                    ],
                },
                "action": {"type": "ContinuousAction"},
                "parking_seed": None,
                "curb_collision_penalty": -10,
                "car_collision_penalty": -20,
                "parking_score_weights": [
                    0.5,
                    0.5,
                    1,
                    1,
                    3,
                ],  # "x", "y", "vx", "vy", "alignment"
                "parking_score_threshold": 0.7,
                "parking_reward": 10,
                "route_following_reward_scalar": 0.1,
                "timestep_penalty": -0.01,
                "max_timesteps": 1000,
                "lane_partition_gridsize": 100,
            }
        )
        return config

    def define_spaces(self) -> None:
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reset(self) -> None:
        self.lanes = self._make_road(self.lanes)

        rng = np.random.default_rng(self.config["parking_seed"])

        self.create_parking_spots(num_spots=2, spot_width=3, spot_height=6, rng=rng)
        spawn_spot = self.road.objects[0]

        self.vehicle = self.action_type.vehicle_class(
            self.road, spawn_spot.position, spawn_spot.heading, 0.0
        )
        self.vehicle.goal = self.road.objects[1]

        self.road.vehicles.append(self.vehicle)

        self.vehicle.parked = False

    def _reward(self, action: Action) -> float:
        """
        Rewards:
            * Curb collision penalty
            * Vehicle-vehicle collision penalty
            * Parking reward (one-time)
            * Timestep punishment
            * Route-following reward
        """
        # Collision
        collided_with_curb = self.detect_object_lane_collision(self.vehicle)
        collided_with_car = self.vehicle.crashed

        if collided_with_curb or collided_with_car:
            self.vehicle.crashed = True
            total_timestep_punishment_left = min(
                (self.config["max_timesteps"] + 1 - self.time)
                * self.config["timestep_penalty"],
                0,
            )

            if collided_with_curb:
                return (
                    self.config["curb_collision_penalty"]
                    + total_timestep_punishment_left
                )
            if collided_with_car:
                return (
                    self.config["car_collision_penalty"]
                    + total_timestep_punishment_left
                )

        # Parking
        parking_score = self.compute_parking_score()
        if parking_score < self.config["parking_score_threshold"]:
            self.vehicle.parked = True
            return self.config["parking_reward"]

        # Route-following
        reward_earned = self.config["timestep_penalty"]

        if self.config["route_following_reward_scalar"] != 0:
            waypoint_vector = (
                self.observation_type.observation_types[0].waypoint
                - self.vehicle.position
            )
            route_following_score = (
                self.config["route_following_reward_scalar"]
                * np.dot(waypoint_vector, self.vehicle.velocity)
                / np.linalg.norm(waypoint_vector)
            )
            # print("Route following score:", route_following_score)
            reward_earned += route_following_score

        return reward_earned

    def _is_terminated(self) -> bool:
        """
        Termination occurs either by collision or by successfully parking
        """
        return self.vehicle.parked or self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time > self.config["max_timesteps"]

    def _info(self, obs: Observation, action: Action | None = None) -> dict:
        info = super()._info(obs, action)
        info["parked"] = self.vehicle.parked
        return info

    def compute_parking_score(self, p: float = 0.5) -> float:
        # We do not use our RelativeGoalObservation to compute reward.
        # Instead we use something similar to compute_reward in ParkingEnv
        # Lower parking score = better

        position_diff = self.vehicle.position - self.vehicle.goal.position
        velocity_diff = self.vehicle.velocity
        alignment_penalty = 1 - abs(
            np.cos(self.vehicle.heading - self.vehicle.goal.heading)
        )  # 0 when perfectly aligned (forward or backward), 1 when sideways

        components = np.array(
            [
                position_diff[0],  # x
                position_diff[1],  # y
                velocity_diff[0],  # vx
                velocity_diff[1],  # vy
                alignment_penalty,  # alignment
            ]
        )
        weights = np.array(self.config["parking_score_weights"])
        return np.power(np.dot(np.abs(components), weights), p)

    def _make_road(self, lanes: list[Lane] | None = None) -> list[Lane]:
        if lanes is None:
            try:
                lanes = generate_random_lanes(self.generation_params)
            except Exception as e:
                raise RuntimeError(
                    "Fatal error encountered when generating road network."
                    "If this issue persists, try a different seed."
                    f"\n\tOriginal error: {e}"
                ) from e

        net = PartitionedRoadNetwork(
            partition_gridsize=self.config["lane_partition_gridsize"]
        )
        for lane in lanes:
            real_lane = PolyLane(
                lane_points=lane.points,
                left_boundary_points=lane.left_points,
                right_boundary_points=lane.right_points,
                line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
            )
            net.add_lane(lane.start, lane.end, real_lane, bidirectional=True)

        self.road = Road(net)

        return lanes

    def create_parking_spots(
        self,
        num_spots: int,
        spot_width: float,
        spot_height: float,
        rng: np.random.Generator,
    ) -> bool:
        """
        :param num_spots: number of parking spots to generate
        :param spot_width: width of parking spot
        [must be less than the lane_width]
        :param spot_height: length of parking spot
        [must be less than forward_speed]
        :param rng: random number generator
        :return: whether or not there was space to generate
        the specified number of spots
        """
        curb_spot_offset = 0.1
        # segment_index: {laneID, side, pt_id (1-(len-2))}
        segment_indices = []

        for id, lane in enumerate(self.lanes):
            for side in ["left_points", "right_points"]:
                for pt_id in range(1, len(getattr(lane, side)) - 2):
                    segment_indices.append({"laneID": id, "side": side, "pt_id": pt_id})

        rng.shuffle(segment_indices)

        num_parking_spots = 0
        segment_indices_i = 0
        while num_parking_spots < num_spots and segment_indices_i < len(
            segment_indices
        ):
            segment_index = segment_indices[segment_indices_i]
            laneID = segment_index["laneID"]
            side = segment_index["side"]
            pt_id = segment_index["pt_id"]

            lane = self.lanes[laneID]
            lane_side = getattr(lane, side)
            pt0 = lane_side[pt_id]
            pt1 = lane_side[pt_id + 1]

            # We will attempt to place a parking spot parallel
            # to our lane segment

            # Requirement 1: This segment must be long enough
            # to encompass the parking spot
            seg_dist = np.linalg.norm(pt0 - pt1)
            if seg_dist < spot_height:
                segment_indices_i += 1
                continue

            # Computing geometry for new parking spot
            vec = pt1 - pt0
            vec /= np.linalg.norm(vec)

            if side == "right_points":
                perp_vec = np.array([vec[1], -vec[0]])
            else:
                perp_vec = np.array([-vec[1], vec[0]])

            center = (pt0 + pt1) / 2 + (perp_vec * (curb_spot_offset + spot_width / 2))
            heading = np.atan2(vec[1], vec[0])

            new_parking_spot = ParkingSpot(self.road, center, heading)
            self.road.objects.append(new_parking_spot)
            num_parking_spots += 1

            # Requirement 2: The rectangular parking space should not
            # intersect with any other lane
            if self.detect_object_lane_collision(new_parking_spot):
                self.road.objects.remove(new_parking_spot)
                num_parking_spots -= 1
                segment_indices_i += 1
                continue

            # Requirement 3: The rectangular parking space should not
            # intersect with any other already existing parking spot
            collision_detected = False
            for other_object in self.road.objects:
                if other_object is not new_parking_spot:
                    collision_detected, _, _ = new_parking_spot._is_colliding(
                        other_object, 0
                    )
                    if collision_detected:
                        break

            if collision_detected:
                self.road.objects.remove(new_parking_spot)
                num_parking_spots -= 1
                segment_indices_i += 1
                continue

            segment_indices_i += 1

        if num_parking_spots < num_spots:
            print(
                "INSUFFICIENT SPOTS FOUND;"
                f" {num_parking_spots} / {num_spots} parking spots generated"
            )
            return False
        return True

    def detect_object_lane_collision(self, object: RoadObject) -> bool:
        gridpoints = set()
        for pt in object.polygon():
            gridpoints.add(point_to_gridpoint(pt, self.road.network.partition_gridsize))

        proximal_lanes = set()
        for gpt in gridpoints:
            proximal_lanes.update(
                get_proximal_lanes_wrt_gridpoint(self.road.network.grid_to_lanes, gpt)
            )

        for lane_index in proximal_lanes:
            lane = self.road.network.get_lane(lane_index)

            left_pairs = zip(lane.left_boundary_points, lane.left_boundary_points[1:])
            right_pairs = zip(
                lane.right_boundary_points, lane.right_boundary_points[1:]
            )

            for p0, p1 in chain(left_pairs, right_pairs):
                if object.intersects_with_line(p0, p1):
                    return True

        return False
