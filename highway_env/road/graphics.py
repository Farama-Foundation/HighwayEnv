import numpy as np
import pygame

from highway_env.road.lane import LineType
from highway_env.vehicle.graphics import VehicleGraphics


class LaneGraphics(object):
    """
        A visualization of a lane.
    """
    STRIPE_SPACING = 5
    """ Offset between stripes [m]"""

    STRIPE_LENGTH = 3
    """ Length of a stripe [m]"""

    STRIPE_WIDTH = 0.3
    """ Width of a stripe [m]"""

    @classmethod
    def display(cls, lane, surface):
        """
            Display a lane on a surface.

        :param lane: the lane to be displayed
        :param surface: the pygame surface
        """
        stripes_count = int(2 * (surface.get_height() + surface.get_width()) / (cls.STRIPE_SPACING * surface.scaling))
        s_origin, _ = lane.local_coordinates(surface.origin)
        s0 = (int(s_origin) // cls.STRIPE_SPACING - stripes_count // 2) * cls.STRIPE_SPACING
        for side in range(2):
            if lane.line_types[side] == LineType.STRIPED:
                cls.striped_line(lane, surface, stripes_count, s0, side)
            elif lane.line_types[side] == LineType.CONTINUOUS:
                cls.continuous_curve(lane, surface, stripes_count, s0, side)
            elif lane.line_types[side] == LineType.CONTINUOUS_LINE:
                cls.continuous_line(lane, surface, stripes_count, s0, side)

    @classmethod
    def striped_line(cls, lane, surface, stripes_count, s0, side):
        """
            Draw a striped line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes to draw
        :param s0: the longitudinal position of the first stripe [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        starts = s0 + np.arange(stripes_count) * cls.STRIPE_SPACING
        ends = s0 + np.arange(stripes_count) * cls.STRIPE_SPACING + cls.STRIPE_LENGTH
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_curve(cls, lane, surface, stripes_count, s0, side):
        """
            Draw a striped line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes to draw
        :param s0: the longitudinal position of the first stripe [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        starts = s0 + np.arange(stripes_count) * cls.STRIPE_SPACING
        ends = s0 + np.arange(stripes_count) * cls.STRIPE_SPACING + cls.STRIPE_SPACING
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_line(cls, lane, surface, stripes_count, s0, side):
        """
            Draw a continuous line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes that would be drawn if the line was striped
        :param s0: the longitudinal position of the start of the line [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        starts = [s0 + 0 * cls.STRIPE_SPACING]
        ends = [s0 + stripes_count * cls.STRIPE_SPACING + cls.STRIPE_LENGTH]
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def draw_stripes(cls, lane, surface, starts, ends, lats):
        """
            Draw a set of stripes along a lane.

        :param lane: the lane
        :param surface: the surface to draw on
        :param starts: a list of starting longitudinal positions for each stripe [m]
        :param ends:  a list of ending longitudinal positions for each stripe [m]
        :param lats: a list of lateral positions for each stripe [m]
        """
        starts = np.clip(starts, 0, lane.length)
        ends = np.clip(ends, 0, lane.length)
        for k in range(len(starts)):
            if abs(starts[k] - ends[k]) > 0.5 * cls.STRIPE_LENGTH:
                pygame.draw.line(surface, surface.WHITE,
                                 (surface.vec2pix(lane.position(starts[k], lats[k]))),
                                 (surface.vec2pix(lane.position(ends[k], lats[k]))),
                                 max(surface.pix(cls.STRIPE_WIDTH), 1))


class RoadGraphics(object):
    """
        A visualization of a road lanes and vehicles.
    """
    @staticmethod
    def display(road, surface):
        """
            Display the road lanes on a surface.

        :param road: the road to be displayed
        :param surface: the pygame surface
        """
        surface.fill(surface.GREY)
        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for l in road.network.graph[_from][_to]:
                    LaneGraphics.display(l, surface)

    @staticmethod
    def display_traffic(road, surface, simulation_frequency=15, offscreen=False):
        """
            Display the road vehicles on a surface.

        :param road: the road to be displayed
        :param surface: the pygame surface
        :param simulation_frequency: simulation frequency
        """
        if road.record_history:
            for v in road.vehicles:
                VehicleGraphics.display_history(v, surface, simulation=simulation_frequency, offscreen=offscreen)
        for v in road.vehicles:
            VehicleGraphics.display(v, surface, offscreen=offscreen)

    @staticmethod
    def display_obstacles(road, surface, offscreen=False):
        """
            Display the obstacles on a surface.

        :param road: the road to be displayed
        :param surface: the pygame surface
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for o in road.obstacles:
            VehicleGraphics.display(o, surface, offscreen=offscreen)


class WorldSurface(pygame.Surface):
    """
        A pygame Surface implementing a local coordinate system so that we can move and zoom in the displayed area.
    """
    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)
    INITIAL_SCALING = 5.5
    INITIAL_CENTERING = [0.5, 0.5]
    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1

    def __init__(self, size, flags, surf):
        super().__init__(size, flags, surf)
        self.origin = np.array([0, 0])
        self.scaling = self.INITIAL_SCALING
        self.centering_position = self.INITIAL_CENTERING

    def pix(self, length):
        """
            Convert a distance [m] to pixels [px].

        :param length: the input distance [m]
        :return: the corresponding size [px]
        """
        return int(length * self.scaling)

    def pos2pix(self, x, y):
        """
            Convert two world coordinates [m] into a position in the surface [px]

        :param x: x world coordinate [m]
        :param y: y world coordinate [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])

    def vec2pix(self, vec):
        """
             Convert a world position [m] into a position in the surface [px].
        :param vec: a world position [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pos2pix(vec[0], vec[1])

    def move_display_window_to(self, position):
        """
            Set the origin of the displayed area to center on a given world position.
        :param position: a world position [m]
        """
        self.origin = position - np.array(
            [self.centering_position[0] * self.get_width() / self.scaling,
             self.centering_position[1] * self.get_height() / self.scaling])

    def handle_event(self, event):
        """
            Handle pygame events for moving and zooming in the displayed area.

        :param event: a pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                self.scaling *= 1 / self.SCALING_FACTOR
            if event.key == pygame.K_o:
                self.scaling *= self.SCALING_FACTOR
            if event.key == pygame.K_m:
                self.centering_position[0] -= self.MOVING_FACTOR
            if event.key == pygame.K_k:
                self.centering_position[0] += self.MOVING_FACTOR
