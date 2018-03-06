from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import numpy as np
from highway.vehicle.dynamics import Vehicle


class AbstractLane(object):
    """
        A lane on the road, described by its central curve.
    """
    metaclass__ = ABCMeta

    @abstractmethod
    def position(self, longitudinal, lateral):
        """
            Convert local lane coordinates to a world position.

        :param longitudinal: longitudinal lane coordinate [m]
        :param lateral: lateral lane coordinate [m]
        :return: the corresponding world position [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def local_coordinates(self, position):
        """
            Convert a world position to local lane coordinates.

        :param position: a world position [m]
        :return: the (longitudinal, lateral) lane coordinates [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def heading_at(self, longitudinal):
        """
            Get the lane heading at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    @abstractmethod
    def width_at(self, longitudinal):
        """
            Get the lane width at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane width [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def is_reachable_from(self, position):
        """
            Whether the lane is reachable from a given world position

        :param position: the world position [m]
        :return: is the lane reachable?
        """
        raise NotImplementedError()

    def on_lane(self, position, longitudinal=None, lateral=None):
        """
            Whether a given world position is on the lane.

        :param position: a world position [m]
        :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if known [m]
        :param lateral: (optional) the corresponding lateral lane coordinate, if known [m]
        :return: is the position on the lane?
        """
        NotImplementedError()


class LineType:
    """
        A lane side line type.
    """
    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2


class StraightLane(AbstractLane):
    """
        A lane going in straight line.
    """
    def __init__(self, origin, heading, width, line_types=None, bounds=None):
        """
            New straight lane.

        :param origin: the lane starting position [m]
        :param heading: the lane direction [rad]
        :param width: the lane width [m]
        :param line_types: the type of lines on both sides of the lane
        :param bounds: longitudinal coordinates of the lane start and end [m]
        """
        super(StraightLane, self).__init__()
        self.bounds = bounds or [-np.inf, np.inf]
        self.origin = origin
        self.heading = heading
        self.width = width
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.direction = np.array([np.cos(self.heading), np.sin(self.heading)])
        self.direction_lateral = np.array([-self.direction[1], self.direction[0]])

    def position(self, longitudinal, lateral):
        return self.origin + longitudinal * self.direction + lateral * self.direction_lateral

    def heading_at(self, s):
        return self.heading

    def width_at(self, s):
        return self.width

    def local_coordinates(self, position):
        delta = position - self.origin
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)
        return longitudinal, lateral

    def on_lane(self, position, longitudinal=None, lateral=None):
        if not longitudinal or not lateral:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = np.abs(lateral) <= self.width_at(longitudinal) / 2 and \
            self.bounds[0] <= longitudinal < self.bounds[1] + Vehicle.LENGTH
        return is_on

    def is_reachable_from(self, position):
        longitudinal, lateral = self.local_coordinates(position)
        is_close = np.abs(lateral) <= 2 * self.width_at(longitudinal) and self.bounds[0] <= longitudinal < self.bounds[
            1]
        return is_close


class SineLane(StraightLane):
    """
        A sinusoidal lane
    """

    def __init__(self, origin, heading, width, amplitude, pulsation, phase, line_types=None, bounds=None):
        """
            New sinusoidal lane.

        :param origin: the lane starting position [m]
        :param heading: the lane axis direction [rad]
        :param width: the lane width [m]
        :param amplitude: the lane oscillation amplitude [m]
        :param pulsation: the lane pulsation [rad/m]
        :param phase: the lane initial phase [rad]
        :param line_types: the type of lines on both sides of the lane
        :param bounds: longitudinal coordinates of the lane start and end [m]
        """
        super(SineLane, self).__init__(origin, heading, width, line_types, bounds)
        self.amplitude = amplitude
        self.pulsation = pulsation
        self.phase = phase

    def position(self, longitudinal, lateral):
        return super(SineLane, self).position(longitudinal, lateral
                                              + self.amplitude * np.sin(self.pulsation * longitudinal + self.phase))

    def heading_at(self, s):
        return super(SineLane, self).heading_at(s) + np.arctan(
            self.amplitude * self.pulsation * np.cos(self.pulsation * s + self.phase))

    def local_coordinates(self, position):
        longitudinal, lateral = super(SineLane, self).local_coordinates(position)
        return longitudinal, lateral - self.amplitude * np.sin(self.pulsation * longitudinal + self.phase)


class LanesConcatenation(AbstractLane):
    """
        A lane defined as the concatenation of several sub-lanes.
    """
    def __init__(self, lanes):
        """
            New concatenated lane.

        :param lanes: the list of lanes composing the concatenated lane
        """
        super(LanesConcatenation, self).__init__()
        self.lanes = lanes

    def segment_from_longitudinal(self, longitudinal):
        """
            Get the index of the sub-lane corresponding to a longitudinal coordinate in the concatenated lane frame.

        :param longitudinal: the longitudinal coordinate in the concatenated lane local frame. [m]
        :return: the index of the sub-lane closest to this position.
        """
        segment = 0
        segment_longitudinal = longitudinal
        for i in range(len(self.lanes) - 1):
            if self.lanes[i].bounds[1] > segment_longitudinal:
                break
            else:
                segment = i + 1
                segment_longitudinal -= self.lanes[i].bounds[1]
        return segment, segment_longitudinal

    def segment_from_position(self, position):
        """
            Get the index of the sub-lane corresponding to world position.

        :param position: a world position [m]
        :return: the index of the sub-lane closest to this position.
        """
        y_min = None
        segment = None
        first_infinite_segment = None
        for i in range(len(self.lanes)):
            if first_infinite_segment is None and not np.isfinite(self.lanes[i].bounds[1]):
                first_infinite_segment = i

            x, y = self.lanes[i].local_coordinates(position)
            if (x > -5 or i == 0) and (x < self.lanes[i].bounds[1] or i == len(self.lanes) - 1):
                if y_min is None or abs(y) < y_min:
                    y_min = abs(y)
                    segment = i
        if first_infinite_segment is not None:
            segment = min(segment, first_infinite_segment)
        return segment

    def position(self, s, lateral):
        segment, segment_longitudinal = self.segment_from_longitudinal(s)
        return self.lanes[segment].position(segment_longitudinal, lateral)

    def heading_at(self, s):
        segment, segment_longitudinal = self.segment_from_longitudinal(s)
        return self.lanes[segment].heading_at(segment_longitudinal)

    def width_at(self, s):
        segment, segment_longitudinal = self.segment_from_longitudinal(s)
        return self.lanes[segment].width_at(segment_longitudinal)

    def on_lane(self, position, longitudinal=None, lateral=None):
        segment = self.segment_from_position(position)
        return self.lanes[segment].on_lane(position)

    def is_reachable_from(self, position):
        segment = self.segment_from_position(position)
        return self.lanes[segment].is_reachable_from(position)

    def local_coordinates(self, position):
        segment = self.segment_from_position(position)
        x, y = self.lanes[segment].local_coordinates(position)
        x += np.sum([self.lanes[i].bounds[1] for i in range(segment)])

        return x, y
