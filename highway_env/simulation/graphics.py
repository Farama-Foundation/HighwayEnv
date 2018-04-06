from __future__ import division, print_function

import numpy as np
import datetime
import pygame
import shutil
import os

from highway_env.agent.graphics import AgentGraphics
from highway_env.road.graphics import WindowSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics


class SimulationWindow(object):
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 150

    VIDEO_SPEED = 2
    OUT_FOLDER = 'out'
    TMP_FOLDER = os.path.join(OUT_FOLDER, 'tmp')

    def __init__(self, simulation=None, agent_displayed=True, record_video=True):
        self.simulation = simulation
        self.agent_displayed = agent_displayed
        self.record_video = record_video

        pygame.init()
        pygame.display.set_caption("Highway-env")

        if self.agent_displayed:
            self.SCREEN_HEIGHT *= 2
            panel_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT / 2)
            self.agent_surface = pygame.Surface(panel_size)
        else:
            panel_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        self.sim_surface = WindowSurface(panel_size, 0, pygame.Surface(panel_size))
        self.clock = pygame.time.Clock()

        if self.record_video:
            self.make_video_dir()
            self.video_name = 'highway_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

    def process(self):
        self.handle_events()
        self.simulation.process()
        self.display()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.simulation.done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.simulation.pause = not self.simulation.pause
            self.sim_surface.handle_event(event)
            if self.simulation.vehicle:
                VehicleGraphics.handle_event(self.simulation.vehicle, event)

    def display(self):
        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.simulation.road, self.sim_surface)
        if self.simulation.planned_trajectory:
            VehicleGraphics.display_trajectory(self.simulation.planned_trajectory, self.sim_surface)
        RoadGraphics.display_traffic(self.simulation.road, self.sim_surface)
        self.screen.blit(self.sim_surface, (0, 0))

        if self.agent_displayed and self.simulation.agent:
            AgentGraphics.display(self.simulation.agent, self.agent_surface)
            self.screen.blit(self.agent_surface, (0, self.SCREEN_HEIGHT / 2))
        self.clock.tick(self.simulation.SIMULATION_FREQUENCY)
        pygame.display.flip()

        if self.record_video and not self.simulation.pause:
            pygame.image.save(self.screen, "{}/{}_{:04d}.bmp".format(self.TMP_FOLDER,
                                                                     self.video_name,
                                                                     self.simulation.t))

    def window_position(self):
        if self.simulation.vehicle:
            return self.simulation.vehicle.position
        else:
            return np.array([0, len(self.road.lanes) / 2 * 4])

    def make_video_dir(self):
        if not os.path.exists(self.OUT_FOLDER):
            os.mkdir(self.OUT_FOLDER)
        self.clear_video_dir()
        os.mkdir(self.TMP_FOLDER)

    def clear_video_dir(self):
        if os.path.exists(self.TMP_FOLDER):
            shutil.rmtree(self.TMP_FOLDER, ignore_errors=True)

    @property
    def done(self):
        return self.simulation.done

    def quit(self):
        pygame.quit()
        if self.record_video:
            os.system("ffmpeg -r {3} -i {0}/{2}_%04d.bmp -vcodec libx264 -crf 25 {1}/{2}.avi"
                      .format(self.TMP_FOLDER,
                              self.OUT_FOLDER,
                              self.video_name,
                              self.VIDEO_SPEED*self.simulation.SIMULATION_FREQUENCY))
            delay = int(np.round(100/(self.VIDEO_SPEED*self.simulation.SIMULATION_FREQUENCY)))
            os.system("convert -delay {3} -loop 0 {0}/{2}*.bmp {1}/{2}.gif"
                      .format(self.TMP_FOLDER,
                              self.OUT_FOLDER,
                              self.video_name,
                              delay))
            self.clear_video_dir()
