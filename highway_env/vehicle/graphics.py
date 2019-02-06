from __future__ import division, print_function
import numpy as np
import pygame

from highway_env.vehicle.dynamics import Vehicle, Obstacle
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle


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
    def display(cls, vehicle, surface, transparent=False):
        """
            Display a vehicle on a pygame surface.

            The vehicle is represented as a colored rotated rectangle.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param transparent: whether the vehicle should be drawn slightly transparent
        """
        v = vehicle
        s = pygame.Surface((surface.pix(v.LENGTH), surface.pix(v.LENGTH)), pygame.SRCALPHA)  # per-pixel alpha
        rect = (0, surface.pix(v.LENGTH) / 2 - surface.pix(v.WIDTH) / 2, surface.pix(v.LENGTH), surface.pix(v.WIDTH))
        pygame.draw.rect(s, cls.get_color(v, transparent), rect, 0)
        pygame.draw.rect(s, cls.BLACK, rect, 1)
        s = pygame.Surface.convert_alpha(s)
        h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        sr = pygame.transform.rotate(s, -h * 180 / np.pi)
        surface.blit(sr, (surface.pos2pix(v.position[0] - v.LENGTH / 2, v.position[1] - v.LENGTH / 2)))

    @classmethod
    def display_trajectory(cls, states, surface):
        """
            Display the whole trajectory of a vehicle on a pygame surface.

        :param states: the list of vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        """
        for vehicle in states:
            cls.display(vehicle, surface, transparent=True)

    @classmethod
    def get_color(cls, vehicle, transparent=False):
        color = cls.DEFAULT_COLOR
        if vehicle.crashed:
            color = cls.RED

        elif isinstance(vehicle, LinearVehicle):
            color = cls.YELLOW
        elif isinstance(vehicle, IDMVehicle):
            color = cls.BLUE
        elif isinstance(vehicle, MDPVehicle):
            color = cls.EGO_COLOR
        elif isinstance(vehicle, Obstacle):
            color = cls.GREEN
        if transparent:
            color = (color[0], color[1], color[2], 50)
        return color

    @classmethod
    def handle_event(cls, vehicle, event):
        """
            Handle a pygame event depending on the vehicle type

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        if isinstance(vehicle, ControlledVehicle):
            cls.control_event(vehicle, event)
        elif isinstance(vehicle, Vehicle):
            cls.dynamics_event(vehicle, event)

    @classmethod
    def control_event(cls, vehicle, event):
        """
            Map the pygame keyboard events to control decisions

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                vehicle.act("FASTER")
            if event.key == pygame.K_LEFT:
                vehicle.act("SLOWER")
            if event.key == pygame.K_DOWN:
                vehicle.act("LANE_RIGHT")
            if event.key == pygame.K_UP:
                vehicle.act("LANE_LEFT")

    @classmethod
    def dynamics_event(cls, vehicle, event):
        """
            Map the pygame keyboard events to dynamics actuation

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        action = vehicle.action.copy()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                action['steering'] = 45 * np.pi / 180
            if event.key == pygame.K_LEFT:
                action['steering'] = -45 * np.pi / 180
            if event.key == pygame.K_DOWN:
                action['acceleration'] = -6
            if event.key == pygame.K_UP:
                action['acceleration'] = 5
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT:
                action['steering'] = 0
            if event.key == pygame.K_LEFT:
                action['steering'] = 0
            if event.key == pygame.K_DOWN:
                action['acceleration'] = 0
            if event.key == pygame.K_UP:
                action['acceleration'] = 0
        if action != vehicle.action:
            vehicle.act(action)
