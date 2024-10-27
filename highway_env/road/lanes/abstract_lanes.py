from __future__ import annotations

from abc import ABCMeta, abstractmethod, ABC

import numpy as np

from highway_env import utils
from highway_env.road.spline import LinearSpline2D
from highway_env.utils import Vector, class_from_path, get_class_path, wrap_to_pi

from highway_env.road.lanes.lane_utils import LineType


class AbstractLane:
    """A lane on the road, described by its central curve."""

    metaclass__ = ABCMeta
    DEFAULT_WIDTH: float = 4
    VEHICLE_LENGTH: float = 5
    length: float = 0
    line_types: list[LineType]

    @abstractmethod
    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        """
        Convert local lane coordinates to a world position.

        :param longitudinal: longitudinal lane coordinate [m]
        :param lateral: lateral lane coordinate [m]
        :return: the corresponding world position [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        """
        Convert a world position to local lane coordinates.

        :param position: a world position [m]
        :return: the (longitudinal, lateral) lane coordinates [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def heading_at(self, longitudinal: float) -> float:
        """
        Get the lane heading at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    @abstractmethod
    def width_at(self, longitudinal: float) -> float:
        """
        Get the lane width at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane width [m]
        """
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: dict):
        """
        Create lane instance from config

        :param config: json dict with lane parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def to_config(self) -> dict:
        """
        Write lane parameters to dict which can be serialized to json

        :return: dict of lane parameters
        """
        raise NotImplementedError()

    def on_lane(
            self,
            position: np.ndarray,
            longitudinal: float = None,
            lateral: float = None,
            margin: float = 0,
    ) -> bool:
        """
        Whether a given world position is on the lane.

        :param position: a world position [m]
        :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if known [m]
        :param lateral: (optional) the corresponding lateral lane coordinate, if known [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        if longitudinal is None or lateral is None:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = (
                np.abs(lateral) <= self.width_at(longitudinal) / 2 + margin
                and -self.VEHICLE_LENGTH <= longitudinal < self.length + self.VEHICLE_LENGTH
        )
        return is_on

    def is_reachable_from(self, position: np.ndarray) -> bool:
        """
        Whether the lane is reachable from a given world position

        :param position: the world position [m]
        :return: is the lane reachable?
        """
        if self.forbidden:
            return False
        longitudinal, lateral = self.local_coordinates(position)
        is_close = (
                np.abs(lateral) <= 2 * self.width_at(longitudinal)
                and 0 <= longitudinal < self.length + self.VEHICLE_LENGTH
        )
        return is_close

    def after_end(
            self, position: np.ndarray, longitudinal: float = None, lateral: float = None
    ) -> bool:
        if not longitudinal:
            longitudinal, _ = self.local_coordinates(position)
        return longitudinal > self.length - self.VEHICLE_LENGTH / 2

    def distance(self, position: np.ndarray):
        """Compute the L1 distance [m] from a position to the lane."""
        s, r = self.local_coordinates(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)

    def distance_with_heading(
            self,
            position: np.ndarray,
            heading: float | None,
            heading_weight: float = 1.0,
    ):
        """Compute a weighted distance in position and heading to the lane."""
        if heading is None:
            return self.distance(position)
        s, r = self.local_coordinates(position)
        angle = np.abs(self.local_angle(heading, s))
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0) + heading_weight * angle

    def local_angle(self, heading: float, long_offset: float):
        """Compute non-normalised angle of heading to the lane."""
        return wrap_to_pi(heading - self.heading_at(long_offset))


class AbstractWeight(ABC):
    weight: int = 0