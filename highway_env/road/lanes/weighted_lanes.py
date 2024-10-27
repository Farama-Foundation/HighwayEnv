from highway_env.road.lanes.abstract_lanes import AbstractWeight, AbstractLane
from highway_env.road.lanes.lane_utils import LineType
from highway_env.road.lanes.unweighted_lanes import StraightLane, SineLane, CircularLane, PolyLaneFixedWidth, PolyLane

from highway_env.utils import Vector

class WeightedStraightLane(StraightLane, AbstractWeight):
    """`StraightLane` with weight"""
    def __init__(
            self,
            start: Vector,
            end: Vector,
            width: float = AbstractLane.DEFAULT_WIDTH,
            line_types: tuple[LineType, LineType] = None,
            forbidden: bool = False,
            speed_limit: float = 20,
            priority: int = 0,
            weight: int = 1,
    ) -> None:
        """
        New straight lane.

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param width: the lane width [m]
        :param line_types: the type of lines on both sides of the lane
        :param forbidden: is changing to this lane forbidden
        :param priority: priority level of the lane, for determining who has right of way
        :param weight: the weight of the lane
        """
        super().__init__(start, end, width, line_types, forbidden, speed_limit, priority)
        self.weight = weight

class WeightedSineLane(SineLane, AbstractWeight):
    """A weighted `SineLane`"""
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
            speed_limit: float = 20,
            priority: int = 0,
            weight: int = 1,
    ) -> None:
        """
        New sinusoidal lane.

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param amplitude: the lane oscillation amplitude [m]
        :param pulsation: the lane pulsation [rad/m]
        :param phase: the lane initial phase [rad]
        :param weight: the lane weight
        """
        super().__init__(start, end, amplitude, pulsation, phase, width, line_types, forbidden, speed_limit, priority)
        self.weight = weight


class WeightedCircularLane(CircularLane, AbstractWeight):
    """A weighted `CircularLane`"""
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
            speed_limit: float = 20,
            priority: int = 0,
            weight: int = 1,
    ) -> None:
        super().__init__(center, radius, start_phase, end_phase, clockwise, width, line_types, forbidden, speed_limit, priority)
        self.weight = weight


class WeightedPolyLaneFixedWidth(PolyLaneFixedWidth, AbstractWeight):
    """Weighted `PolyLaneFixedWidth`"""
    def __init__(
            self,
            lane_points: list[tuple[float, float]],
            width: float = AbstractLane.DEFAULT_WIDTH,
            line_types: tuple[LineType, LineType] = None,
            forbidden: bool = False,
            speed_limit: float = 20,
            priority: int = 0,
            weight: int = 1,
    ) -> None:
        super().__init__(lane_points, width, line_types, forbidden, speed_limit, priority)
        self.weight = weight

class WeightedPolyLane(PolyLane, AbstractWeight):
    """A weighted `PolyLane`"""
    def __init__(
            self,
            lane_points: list[tuple[float, float]],
            left_boundary_points: list[tuple[float, float]],
            right_boundary_points: list[tuple[float, float]],
            line_types: tuple[LineType, LineType] = None,
            forbidden: bool = False,
            speed_limit: float = 20,
            priority: int = 0,
            weight: int = 1,
    ) -> None:
        super().__init__(lane_points, left_boundary_points, right_boundary_points, line_types, forbidden, speed_limit, priority)
        self.weight = weight