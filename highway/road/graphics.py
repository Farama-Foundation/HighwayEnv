from __future__ import division, print_function
import numpy as np
import pygame

from highway.road.lane import LineType, LanesConcatenation
from highway.vehicle.graphics import VehicleGraphics


class LaneGraphics(object):
    STRIPE_SPACING = 5
    STRIPE_LENGTH = 3
    STRIPE_WIDTH = 0.3

    @classmethod
    def display(cls, lane, surface):
        if isinstance(lane, LanesConcatenation):
            for i in range(len(lane.lanes)):
                cls.display(lane.lanes[i], surface)
            return

        stripes_count = int(2 * (surface.get_height() + surface.get_width()) / (cls.STRIPE_SPACING * surface.scaling))
        s_origin, _ = lane.local_coordinates(surface.origin)
        s0 = (int(s_origin) // cls.STRIPE_SPACING - stripes_count // 2) * cls.STRIPE_SPACING
        for side in range(2):
            if lane.line_types[side] == LineType.STRIPED:
                cls.striped_line(lane, surface, stripes_count, s0, side)
            elif lane.line_types[side] == LineType.CONTINUOUS:
                cls.continuous_line(lane, surface, stripes_count, s0, side)

    @classmethod
    def striped_line(cls, lane, surface, stripes_count, s0, side):
        starts = s0 + np.arange(stripes_count) * cls.STRIPE_SPACING
        ends = s0 + np.arange(stripes_count) * cls.STRIPE_SPACING + cls.STRIPE_LENGTH
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_line(cls, lane, surface, stripes_count, s0, side):
        starts = [s0 + 0 * cls.STRIPE_SPACING]
        ends = [s0 + stripes_count * cls.STRIPE_SPACING + cls.STRIPE_LENGTH]
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def draw_stripes(cls, lane, surface, starts, ends, lats):
        starts = np.clip(starts, lane.bounds[0], lane.bounds[1])
        ends = np.clip(ends, lane.bounds[0], lane.bounds[1])
        for k in range(len(starts)):
            if abs(starts[k] - ends[k]) > 0.5 * cls.STRIPE_LENGTH:
                pygame.draw.line(surface, surface.WHITE,
                                 (surface.vec2pix(lane.position(starts[k], lats[k]))),
                                 (surface.vec2pix(lane.position(ends[k], lats[k]))),
                                 max(surface.pix(cls.STRIPE_WIDTH), 1))


class RoadGraphics(object):
    @classmethod
    def display(cls, road, surface):
        surface.fill(surface.GREY)
        for l in road.lanes:
            LaneGraphics.display(l, surface)

    @classmethod
    def display_traffic(cls, road, surface):
        for v in road.vehicles:
            VehicleGraphics.display(v, surface)


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
