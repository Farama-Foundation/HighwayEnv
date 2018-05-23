from __future__ import division, print_function, absolute_import

import numpy as np
import pygame

from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics


class EnvViewer(object):
    """
        A viewer to render a highway driving environment.
    """
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 300

    def __init__(self, env):
        self.env = env

        pygame.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.clock = pygame.time.Clock()

        self.agent_display = None
        self.agent_surface = None

    def set_agent_display(self, agent_display):
        if self.agent_display is None:
            self.agent_display = agent_display
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, 2 * self.SCREEN_HEIGHT))
            self.agent_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

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
        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)
        RoadGraphics.display_traffic(self.env.road, self.sim_surface)
        self.screen.blit(self.sim_surface, (0, 0))

        if self.agent_display:
            self.agent_display(self.agent_surface)
            self.screen.blit(self.agent_surface, (0, self.SCREEN_HEIGHT))

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
            if False:
                return self.env.vehicle.position
            else:
                return np.array([self.env.vehicle.position[0], len(self.env.road.lanes) / 2 * 4 - 2])
        else:
            return np.array([0, len(self.env.road.lanes) / 2 * 4])

    def close(self):
        """
            Close the pygame window.
        """
        pygame.quit()

