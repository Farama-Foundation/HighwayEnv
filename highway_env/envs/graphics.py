from __future__ import division, print_function, absolute_import

import os

import numpy as np
import pygame

from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics


class EnvViewer(object):
    """
        A viewer to render a highway driving environment.
    """
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 150

    def __init__(self, env):
        self.env = env

        pygame.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.centering_position = env.config.get("centering_position", self.sim_surface.INITIAL_CENTERING)
        self.clock = pygame.time.Clock()

        self.enabled = True
        if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
            self.enabled = False

        self.agent_display = None
        self.agent_surface = None

    def set_agent_display(self, agent_display):
        if self.agent_display is None:
            if self.SCREEN_WIDTH > self.SCREEN_HEIGHT:
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, 2 * self.SCREEN_HEIGHT))
            else:
                self.screen = pygame.display.set_mode((2 * self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.agent_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.agent_display = agent_display

    def handle_events(self):
        """
            Handle pygame events by forwarding them to the display and environment vehicle.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.vehicle:
                VehicleGraphics.handle_event(self.env.vehicle, event)

    def display(self):
        """
            Display the road and vehicles on a pygame window.
        """
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)
        RoadGraphics.display_traffic(self.env.road, self.sim_surface)

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if self.SCREEN_WIDTH > self.SCREEN_HEIGHT:
                self.screen.blit(self.agent_surface, (0, self.SCREEN_HEIGHT))
            else:
                self.screen.blit(self.agent_surface, (self.SCREEN_WIDTH, 0))

        self.screen.blit(self.sim_surface, (0, 0))
        self.clock.tick(self.env.SIMULATION_FREQUENCY)
        pygame.display.flip()

    def get_image(self):
        """
        :return: the rendered image as a rbg array
        """
        data = pygame.surfarray.array3d(self.screen)
        return np.moveaxis(data, 0, 1)

    def window_position(self):
        """
        :return: the world position of the center of the displayed window.
        """
        if self.env.vehicle:
            return self.env.vehicle.position
        else:
            return np.array([0, 0])

    def close(self):
        """
            Close the pygame window.
        """
        pygame.quit()

