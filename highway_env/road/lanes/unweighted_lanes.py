from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.road.spline import LinearSpline2D
from highway_env.utils import Vector, class_from_path, get_class_path

from highway_env.road.lanes.abstract_lanes import AbstractLane
from highway_env.road.lanes.lane_utils import LineType


class StraightLane(AbstractLane):
    """A lane going in straight line."""

    def __init__(
        self,
        start: Vector,
        end: Vector,
        width: float = AbstractLane.DEFAULT_WIDTH,
        line_types: tuple[LineType, LineType] = None,
        forbidden: bool = False,
        speed_limit: float = 40,
        priority: int = 0,
    ) -> None:
        """
        New straight lane.

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param width: the lane width [m]
        :param line_types: the type of lines on both sides of the lane
        :param forbidden: is changing to this lane forbidden
        :param priority: priority level of the lane, for determining who has right of way
        """
        self.start = np.array(start)
        self.end = np.array(end)
        self.width = width
        self.heading = np.arctan2(
            self.end[1] - self.start[1], self.end[0] - self.start[0]
        )
        self.length = np.linalg.norm(self.end - self.start)
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.direction = (self.end - self.start) / self.length
        self.direction_lateral = np.array([-self.direction[1], self.direction[0]])
        self.forbidden = forbidden
        self.priority = priority
        self.speed_limit = speed_limit

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return (
            self.start
            + longitudinal * self.direction
            + lateral * self.direction_lateral
        )

    def heading_at(self, longitudinal: float) -> float:
        return self.heading

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        delta = position - self.start
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)
        return float(longitudinal), float(lateral)

    @classmethod
    def from_config(cls, config: dict):
        config["start"] = np.array(config["start"])
        config["end"] = np.array(config["end"])
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_path": get_class_path(self.__class__),
            "config": {
                "start": _to_serializable(self.start),
                "end": _to_serializable(self.end),
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority,
            },
        }


class SineLane(StraightLane):
    """A sinusoidal lane."""

    def __init__(
        self,
        start: Vector,
        end: Vector,
        amplitude: float,
        pulsation: float,
        phase: float,
        width: float = StraightLane.DEFAULT_WIDTH,
        line_types: list[LineType] = None,
        forbidden: bool = False,
        speed_limit: float = 40,
        priority: int = 0,
    ) -> None:
        """
        New sinusoidal lane.

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param amplitude: the lane oscillation amplitude [m]
        :param pulsation: the lane pulsation [rad/m]
        :param phase: the lane initial phase [rad]
        """
        super().__init__(
            start, end, width, line_types, forbidden, speed_limit, priority
        )
        self.amplitude = amplitude
        self.pulsation = pulsation
        self.phase = phase

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return super().position(
            longitudinal,
            lateral
            + self.amplitude * np.sin(self.pulsation * longitudinal + self.phase),
        )

    def heading_at(self, longitudinal: float) -> float:
        return super().heading_at(longitudinal) + np.arctan(
            self.amplitude
            * self.pulsation
            * np.cos(self.pulsation * longitudinal + self.phase)
        )

    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        longitudinal, lateral = super().local_coordinates(position)
        return longitudinal, lateral - self.amplitude * np.sin(
            self.pulsation * longitudinal + self.phase
        )

    @classmethod
    def from_config(cls, config: dict):
        config["start"] = np.array(config["start"])
        config["end"] = np.array(config["end"])
        return cls(**config)

    def to_config(self) -> dict:
        config = super().to_config()
        config.update(
            {
                "class_path": get_class_path(self.__class__),
            }
        )
        config["config"].update(
            {
                "amplitude": self.amplitude,
                "pulsation": self.pulsation,
                "phase": self.phase,
            }
        )
        return config


class CircularLane(AbstractLane):
    """A lane going in circle arc."""

    def __init__(
        self,
        center: Vector,
        radius: float,
        start_phase: float,
        end_phase: float,
        clockwise: bool = True,
        width: float = AbstractLane.DEFAULT_WIDTH,
        line_types: list[LineType] = None,
        forbidden: bool = False,
        speed_limit: float = 40,
        priority: int = 0,
    ) -> None:
        super().__init__()
        self.center = np.array(center)
        self.radius = radius
        self.start_phase = start_phase
        self.end_phase = end_phase
        self.clockwise = clockwise
        self.direction = 1 if clockwise else -1
        self.width = width
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.forbidden = forbidden
        self.length = radius * (end_phase - start_phase) * self.direction
        self.priority = priority
        self.speed_limit = speed_limit

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        return self.center + (self.radius - lateral * self.direction) * np.array(
            [np.cos(phi), np.sin(phi)]
        )

    def heading_at(self, longitudinal: float) -> float:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        psi = phi + np.pi / 2 * self.direction
        return psi

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        delta = position - self.center
        phi = np.arctan2(delta[1], delta[0])
        phi = self.start_phase + utils.wrap_to_pi(phi - self.start_phase)
        r = np.linalg.norm(delta)
        longitudinal = self.direction * (phi - self.start_phase) * self.radius
        lateral = self.direction * (self.radius - r)
        return longitudinal, lateral

    @classmethod
    def from_config(cls, config: dict):
        config["center"] = np.array(config["center"])
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_path": get_class_path(self.__class__),
            "config": {
                "center": _to_serializable(self.center),
                "radius": self.radius,
                "start_phase": self.start_phase,
                "end_phase": self.end_phase,
                "clockwise": self.clockwise,
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority,
            },
        }


class PolyLaneFixedWidth(AbstractLane):
    """
    A fixed-width lane defined by a set of points and approximated with a 2D Hermite polynomial.
    """

    def __init__(
        self,
        lane_points: list[tuple[float, float]],
        width: float = AbstractLane.DEFAULT_WIDTH,
        line_types: tuple[LineType, LineType] = None,
        forbidden: bool = False,
        speed_limit: float = 40,
        priority: int = 0,
    ) -> None:
        self.curve = LinearSpline2D(lane_points)
        self.length = self.curve.length
        self.width = width
        self.line_types = line_types
        self.forbidden = forbidden
        self.speed_limit = speed_limit
        self.priority = priority

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        x, y = self.curve(longitudinal)
        yaw = self.heading_at(longitudinal)
        return np.array([x - np.sin(yaw) * lateral, y + np.cos(yaw) * lateral])

    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        lon, lat = self.curve.cartesian_to_frenet(position)
        return lon, lat

    def heading_at(self, longitudinal: float) -> float:
        dx, dy = self.curve.get_dx_dy(longitudinal)
        return np.arctan2(dy, dx)

    def width_at(self, longitudinal: float) -> float:
        return self.width

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "config": {
                "lane_points": _to_serializable(
                    [_to_serializable(p.position) for p in self.curve.poses]
                ),
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority,
            },
        }


class PolyLane(PolyLaneFixedWidth):
    """
    A lane defined by a set of points and approximated with a 2D Hermite polynomial.
    """

    def __init__(
        self,
        lane_points: list[tuple[float, float]],
        left_boundary_points: list[tuple[float, float]],
        right_boundary_points: list[tuple[float, float]],
        line_types: tuple[LineType, LineType] = None,
        forbidden: bool = False,
        speed_limit: float = 40,
        priority: int = 0,
    ):
        super().__init__(
            lane_points=lane_points,
            line_types=line_types,
            forbidden=forbidden,
            speed_limit=speed_limit,
            priority=priority,
        )
        self.right_boundary = LinearSpline2D(right_boundary_points)
        self.left_boundary = LinearSpline2D(left_boundary_points)
        self._init_width()

    def width_at(self, longitudinal: float) -> float:
        if longitudinal < 0:
            return self.width_samples[0]
        elif longitudinal > len(self.width_samples) - 1:
            return self.width_samples[-1]
        else:
            return self.width_samples[int(longitudinal)]

    def _width_at_s(self, longitudinal: float) -> float:
        """
        Calculate width by taking the minimum distance between centerline and each boundary at a given s-value. This compensates indentations in boundary lines.
        """
        center_x, center_y = self.position(longitudinal, 0)
        right_x, right_y = self.right_boundary(
            self.right_boundary.cartesian_to_frenet([center_x, center_y])[0]
        )
        left_x, left_y = self.left_boundary(
            self.left_boundary.cartesian_to_frenet([center_x, center_y])[0]
        )

        dist_to_center_right = np.linalg.norm(
            np.array([right_x, right_y]) - np.array([center_x, center_y])
        )
        dist_to_center_left = np.linalg.norm(
            np.array([left_x, left_y]) - np.array([center_x, center_y])
        )

        return max(
            min(dist_to_center_right, dist_to_center_left) * 2,
            AbstractLane.DEFAULT_WIDTH,
        )

    def _init_width(self):
        """
        Pre-calculate sampled width values in about 1m distance to reduce computation during runtime. It is assumed that the width does not change significantly within 1-2m.
        Using numpys linspace ensures that min and max s-values are contained in the samples.
        """
        s_samples = np.linspace(
            0,
            self.curve.length,
            num=int(np.ceil(self.curve.length)) + 1,
        )
        self.width_samples = [self._width_at_s(s) for s in s_samples]

    def to_config(self) -> dict:
        config = super().to_config()

        ordered_boundary_points = _to_serializable(
            [_to_serializable(p.position) for p in reversed(self.left_boundary.poses)]
        )
        ordered_boundary_points += _to_serializable(
            [_to_serializable(p.position) for p in self.right_boundary.poses]
        )

        config["class_name"] = self.__class__.__name__
        config["config"]["ordered_boundary_points"] = ordered_boundary_points
        del config["config"]["width"]

        return config


def _to_serializable(arg: np.ndarray | list) -> list:
    if isinstance(arg, np.ndarray):
        return arg.tolist()
    return arg


def lane_from_config(cfg: dict) -> AbstractLane:
    return class_from_path(cfg["class_path"])(**cfg["config"])
