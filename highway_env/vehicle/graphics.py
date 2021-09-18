import itertools
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import pygame

from highway_env.utils import Vector
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle

if TYPE_CHECKING:
    from highway_env.road.graphics import WorldSurface


class VehicleGraphics(object):
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN

    @classmethod
    def display(cls, vehicle: Vehicle, surface: "WorldSurface",
                transparent: bool = False,
                offscreen: bool = False,
                label: bool = False,
                draw_roof: bool = False) -> None:
        """
        Display a vehicle on a pygame surface.

        The vehicle is represented as a colored rotated rectangle.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param transparent: whether the vehicle should be drawn slightly transparent
        :param offscreen: whether the rendering should be done offscreen or not
        :param label: whether a text label should be rendered
        """
        if not surface.is_visible(vehicle.position):
            return

        v = vehicle
        tire_length, tire_width = 1, 0.3
        headlight_length, headlight_width = 0.72, 0.6
        roof_length, roof_width = 2.0, 1.5

        # Vehicle rectangle
        length = v.LENGTH + 2 * tire_length
        vehicle_surface = pygame.Surface((surface.pix(length), surface.pix(length)),
                                         flags=pygame.SRCALPHA)  # per-pixel alpha
        rect = (surface.pix(tire_length),
                surface.pix(length / 2 - v.WIDTH / 2),
                surface.pix(v.LENGTH),
                surface.pix(v.WIDTH))
        rect_headlight_left = (surface.pix(tire_length+v.LENGTH-headlight_length),
                               surface.pix(length / 2 - (1.4*v.WIDTH) / 3),
                               surface.pix(headlight_length),
                               surface.pix(headlight_width))
        rect_headlight_right = (surface.pix(tire_length+v.LENGTH-headlight_length),
                                surface.pix(length / 2 + (0.6*v.WIDTH) / 5),
                                surface.pix(headlight_length),
                                surface.pix(headlight_width))
        color = cls.get_color(v, transparent)
        pygame.draw.rect(vehicle_surface, color, rect, 0)
        pygame.draw.rect(vehicle_surface, cls.lighten(color), rect_headlight_left, 0)
        pygame.draw.rect(vehicle_surface, cls.lighten(color), rect_headlight_right, 0)
        if draw_roof:
            rect_roof = (surface.pix(v.LENGTH/2 - tire_length/2),
                         surface.pix(0.999*length/ 2 - 0.38625*v.WIDTH),
                         surface.pix(roof_length),
                         surface.pix(roof_width))
            pygame.draw.rect(vehicle_surface, cls.darken(color), rect_roof, 0)
        pygame.draw.rect(vehicle_surface, cls.BLACK, rect, 1)

        # Tires
        if type(vehicle) in [Vehicle, BicycleVehicle]:
            tire_positions = [[surface.pix(tire_length), surface.pix(length / 2 - v.WIDTH / 2)],
                              [surface.pix(tire_length), surface.pix(length / 2 + v.WIDTH / 2)],
                              [surface.pix(length - tire_length), surface.pix(length / 2 - v.WIDTH / 2)],
                              [surface.pix(length - tire_length), surface.pix(length / 2 + v.WIDTH / 2)]]
            tire_angles = [0, 0, v.action["steering"], v.action["steering"]]
            for tire_position, tire_angle in zip(tire_positions, tire_angles):
                tire_surface = pygame.Surface((surface.pix(tire_length), surface.pix(tire_length)), pygame.SRCALPHA)
                rect = (0, surface.pix(tire_length/2-tire_width/2), surface.pix(tire_length), surface.pix(tire_width))
                pygame.draw.rect(tire_surface, cls.BLACK, rect, 0)
                cls.blit_rotate(vehicle_surface, tire_surface, tire_position, np.rad2deg(-tire_angle))

        # Centered rotation
        h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        position = [*surface.pos2pix(v.position[0], v.position[1])]
        if not offscreen:
            # convert_alpha throws errors in offscreen mode
            # see https://stackoverflow.com/a/19057853
            vehicle_surface = pygame.Surface.convert_alpha(vehicle_surface)
        cls.blit_rotate(surface, vehicle_surface, position, np.rad2deg(-h))

        # Label
        if label:
            font = pygame.font.Font(None, 15)
            text = "#{}".format(id(v) % 1000)
            text = font.render(text, 1, (10, 10, 10), (255, 255, 255))
            surface.blit(text, position)

    @staticmethod
    def blit_rotate(surf: pygame.SurfaceType, image: pygame.SurfaceType, pos: Vector, angle: float,
                    origin_pos: Vector = None, show_rect: bool = False) -> None:
        """Many thanks to https://stackoverflow.com/a/54714144."""
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

        # calculate the translation of the pivot
        if origin_pos is None:
            origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (pos[0] - origin_pos[0] + min_box[0] - pivot_move[0], pos[1] - origin_pos[1] - max_box[1] + pivot_move[1])
        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        # draw rectangle around the image
        if show_rect:
            pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)

    @classmethod
    def display_trajectory(cls, states: List[Vehicle], surface: "WorldSurface", offscreen: bool = False) -> None:
        """
        Display the whole trajectory of a vehicle on a pygame surface.

        :param states: the list of vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for vehicle in states:
            cls.display(vehicle, surface, transparent=True, offscreen=offscreen)

    @classmethod
    def display_history(cls, vehicle: Vehicle, surface: "WorldSurface", frequency: float = 3, duration: float = 2,
                        simulation: int = 15, offscreen: bool = False) -> None:
        """
        Display the whole trajectory of a vehicle on a pygame surface.

        :param vehicle: the vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        :param frequency: frequency of displayed positions in history
        :param duration: length of displayed history
        :param simulation: simulation frequency
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for v in itertools.islice(vehicle.history,
                                  None,
                                  int(simulation * duration),
                                  int(simulation / frequency)):
            cls.display(v, surface, transparent=True, offscreen=offscreen)

    @classmethod
    def get_color(cls, vehicle: Vehicle, transparent: bool = False) -> Tuple[int]:
        color = cls.DEFAULT_COLOR
        if getattr(vehicle, "color", None):
            color = vehicle.color
        elif vehicle.crashed:
            color = cls.RED
        elif isinstance(vehicle, LinearVehicle):
            color = cls.YELLOW
        elif isinstance(vehicle, IDMVehicle):
            color = cls.BLUE
        elif isinstance(vehicle, MDPVehicle):
            color = cls.EGO_COLOR
        if transparent:
            color = (color[0], color[1], color[2], 30)
        return color

    @classmethod
    def darken(cls, color, ratio=0.83):
        return (
            int(color[0] * ratio),
            int(color[1] * ratio),
            int(color[2] * ratio),
        ) + color[3:]

    @classmethod
    def lighten(cls, color, ratio=0.68):
        return (
            min(int(color[0] / ratio), 255),
            min(int(color[1] / ratio), 255),
            min(int(color[2] / ratio), 255),
        ) + color[3:]
