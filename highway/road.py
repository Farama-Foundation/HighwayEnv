from __future__ import division, print_function
import numpy as np
import pandas as pd
import pygame
from highway import utils
from highway.logger import Loggable
from highway.vehicle import ControlledVehicle, IDMVehicle, Obstacle, Vehicle


class Lane(object):
    STRIPE_SPACING = 5
    STRIPE_LENGTH = 3
    STRIPE_WIDTH = 0.3

    def __init__(self):
        pass

    def position(self, longitudinal, lateral):
        raise Exception('Not implemented.')

    def heading_at(self, s):
        raise Exception('Not implemented.')

    def width_at(self, s):
        raise Exception('Not implemented.')

    def local_coordinates(self, position):
        raise Exception('Not implemented.')

    def is_reachable_from(self, position):
        raise Exception('Not implemented.')


class LineType:
    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2


class StraightLane(Lane):
    def __init__(self, origin, heading, width, line_types=None, bounds=None):
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

    def display(self, screen):
        stripes_count = int(2 * (screen.get_height() + screen.get_width()) / (self.STRIPE_SPACING * screen.scaling))
        s_origin, _ = self.local_coordinates(screen.origin)
        s0 = (int(s_origin) // self.STRIPE_SPACING - stripes_count // 2) * self.STRIPE_SPACING
        for side in range(2):
            if self.line_types[side] == LineType.STRIPED:
                self.striped_line(screen, stripes_count, s0, side)
            elif self.line_types[side] == LineType.CONTINUOUS:
                self.continuous_line(screen, stripes_count, s0, side)

    def striped_line(self, screen, stripes_count, s0, side):
        starts = s0 + np.arange(stripes_count) * self.STRIPE_SPACING
        ends = s0 + np.arange(stripes_count) * self.STRIPE_SPACING + self.STRIPE_LENGTH
        lat = (side - 0.5) * self.width
        self.draw_stripes(screen, starts, ends, lat)

    def continuous_line(self, screen, stripes_count, s0, side):
        starts = [s0 + 0 * self.STRIPE_SPACING]
        ends = [s0 + stripes_count * self.STRIPE_SPACING + self.STRIPE_LENGTH]
        lat = (side - 0.5) * self.width
        self.draw_stripes(screen, starts, ends, lat)

    def draw_stripes(self, screen, starts, ends, lat):
        starts = np.clip(starts, self.bounds[0], self.bounds[1])
        ends = np.clip(ends, self.bounds[0], self.bounds[1])
        for k in range(len(starts)):
            if abs(starts[k] - ends[k]) > 0.5 * self.STRIPE_LENGTH:
                pygame.draw.line(screen, screen.WHITE,
                                 (screen.vec2pix(self.position(starts[k], lat))),
                                 (screen.vec2pix(self.position(ends[k], lat))),
                                 max(screen.pix(self.STRIPE_WIDTH), 1))


class SineLane(StraightLane):
    STRIPE_SPACING = 5
    STRIPE_LENGTH = 3

    def __init__(self, origin, heading, width, amplitude, pulsation, phase, line_types=None, bounds=None):
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


class LanesConcatenation(Lane):
    def __init__(self, lanes):
        super(LanesConcatenation, self).__init__()
        self.lanes = lanes

    def segment_from_longitudinal(self, longitudinal):
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
        y_min = None
        segment = None
        first_infinite_segment = None
        for i in range(len(self.lanes)):
            if first_infinite_segment is None and not np.isfinite(self.lanes[i].bounds[1]):
                first_infinite_segment = i

            x, y = self.lanes[i].local_coordinates(position)
            if (x > -self.STRIPE_SPACING or i == 0) and (x < self.lanes[i].bounds[1] or i == len(self.lanes) - 1):
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

    def display(self, screen):
        for i in range(len(self.lanes)):
            self.lanes[i].display(screen)


class Road(Loggable):
    """
        The set of vehicles on the road, and its characteristics
    """

    def __init__(self, lanes=None, vehicles=None):
        self.lanes = lanes or []
        self.vehicles = vehicles or []

    @classmethod
    def create_random_road(cls, lanes_count, lane_width, vehicles_count=50, vehicles_type=ControlledVehicle):
        lanes = []
        for lane in range(lanes_count):
            origin = np.array([0, lane * lane_width])
            heading = 0
            line_types = [LineType.CONTINUOUS if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS if lane == lanes_count - 1 else LineType.NONE]
            lanes.append(StraightLane(origin, heading, lane_width, line_types))
        r = Road(lanes)
        r.add_random_vehicles(vehicles_count, vehicles_type)
        return r

    def add_random_vehicles(self, vehicles_count=50, vehicles_type=ControlledVehicle):
        for _ in range(vehicles_count):
            self.vehicles.append(vehicles_type.create_random(self))

    def act(self):
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt):
        for vehicle in self.vehicles:
            vehicle.step(dt)
            for other in self.vehicles:
                vehicle.check_collision(other)

    def get_lane(self, position):
        return self.lanes[self.get_lane_index(position)]

    def get_lane_index(self, position):
        lateral = [abs(l.local_coordinates(position)[1]) for l in self.lanes]
        return int(np.argmin(lateral))

    def neighbour_vehicles(self, vehicle, lane=None):
        lane = lane or vehicle.lane
        if not lane:
            return None, None
        s = lane.local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles:
            if v is not vehicle and v.lane == lane:
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    def display_road(self, screen):
        screen.fill(screen.GREY)
        for l in self.lanes:
            l.display(screen)

    def display_traffic(self, screen):
        for v in self.vehicles:
            v.display(screen)

    def dump(self):
        for v in self.vehicles:
            if not isinstance(v, Obstacle):
                v.dump()

    def get_log(self):
        return pd.concat([v.get_log() for v in self.vehicles])

    def __repr__(self):
        return self.vehicles.__repr__()


class RoadSurface(pygame.Surface):
    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)
    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1

    def __init__(self, size, flags, surf):
        super(RoadSurface, self).__init__(size, flags, surf)
        self.origin = np.array([0, 0])
        self.scaling = 10.0
        self.centering_position = 0.3

    def pix(self, length):
        return int(length * self.scaling)

    def pos2pix(self, x, y):
        return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])

    def vec2pix(self, vec):
        return self.pos2pix(vec[0], vec[1])

    def move_display_window_to(self, position):
        self.origin = position - np.array(
            [self.centering_position * self.get_width() / self.scaling, self.get_height() / (2 * self.scaling)])

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                self.scaling *= 1 / self.SCALING_FACTOR
            if event.key == pygame.K_o:
                self.scaling *= self.SCALING_FACTOR
            if event.key == pygame.K_m:
                self.centering_position -= self.MOVING_FACTOR
            if event.key == pygame.K_k:
                self.centering_position += self.MOVING_FACTOR


def test():
    from highway.simulation import Simulation
    # l = SineLane(sim.road.lanes[-1].origin+np.array([0,9]),0, 4.0, 3, 6.28/60, [False,False])
    ends = [100, 50, 100]
    l0 = StraightLane(np.array([0, 0]), 0, 4.0, [LineType.CONTINUOUS, LineType.NONE])
    l1 = StraightLane(np.array([0, 4]), 0, 4.0, [LineType.STRIPED, LineType.CONTINUOUS])

    lc0 = StraightLane(np.array([0, 6.5 + 4 + 4]), 0, 4.0, [LineType.CONTINUOUS, LineType.CONTINUOUS],
                       bounds=[-np.inf, ends[0]])
    amplitude = 3.3
    lc1 = SineLane(lc0.position(ends[0], -amplitude), 0, 4.0, amplitude, 2 * np.pi / 100, np.pi / 2,
                   [LineType.STRIPED, LineType.STRIPED], bounds=[0, ends[1]])
    lc2 = StraightLane(lc1.position(ends[1], 0), 0, 4.0, [LineType.NONE, LineType.CONTINUOUS],
                       bounds=[0, ends[2]])
    l2 = LanesConcatenation([lc0, lc1, lc2])
    road = Road([l0, l1, l2])
    sim = Simulation(road, ego_vehicle_type=ControlledVehicle)
    road.vehicles.append(Obstacle(road, lc2.position(ends[2], 0)))

    while not sim.done:
        sim.process()
    sim.quit()


if __name__ == '__main__':
    test()
