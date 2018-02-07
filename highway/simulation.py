from __future__ import division, print_function
import pygame
from highway.vehicle import MDPVehicle, IDMVehicle
from highway.road import Road, RoadSurface
from highway.mdp import RoadMDP, SimplifiedMDP
import numpy as np
import os
import logging


class Simulation:
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 600
    FPS = 30
    dt = 1 / 30
    POLICY_FREQUENCY = 1
    RECORD_VIDEO = False
    OUT_FOLDER = 'out'

    def __init__(self, road, ego_vehicle_type=None, displayed=True):
        self.road = road
        if ego_vehicle_type:
            self.vehicle = ego_vehicle_type.create_random(self.road, 25, ego=True)
            self.road.vehicles.append(self.vehicle)
        else:
            self.vehicle = None
        self.displayed = displayed

        self.t = 0
        self.done = False
        self.pause = False
        self.trajectory = None
        self.smdp = None
        if self.displayed:
            pygame.init()
            pygame.display.set_caption("Highway")
            size = [self.SCREEN_WIDTH, self.SCREEN_HEIGHT]
            panel_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT / 2)
            self.screen = pygame.display.set_mode(size)
            self.road_surface = RoadSurface(panel_size, 0, pygame.Surface(panel_size))
            self.value_surface = pygame.Surface(panel_size)
            self.clock = pygame.time.Clock()

    def process(self):
        self.handle_events()
        self.act()
        self.step()
        if self.displayed:
            self.display()

    def handle_events(self):
        if self.displayed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.pause = not self.pause
                self.road_surface.handle_event(event)
                if self.vehicle:
                    self.vehicle.handle_event(event)

    def act(self):
        if self.pause:
            return

        # Default action for all vehicles
        self.road.act()

        # Planning for ego-vehicle
        policy_call = self.t % (self.FPS // self.POLICY_FREQUENCY) == 0
        if self.vehicle and isinstance(self.vehicle, MDPVehicle) and policy_call:
            mdp = RoadMDP(self.road, self.vehicle)
            self.smdp = SimplifiedMDP(mdp.state)
            self.smdp.value_iteration()
            logging.debug(mdp.state)
            logging.debug(self.smdp.value)

            _, actions = self.smdp.plan()
            self.trajectory = self.vehicle.predict_trajectory(actions,
                                                              mdp.TIME_QUANTIFICATION,
                                                              8 * self.dt,
                                                              self.dt)
            action = self.smdp.pick_action()
            logging.debug(actions)
            logging.debug(action)

            self.vehicle.act(action)

    def step(self):
        if not self.pause:
            self.road.step(self.dt)
            self.t += 1

    def display_position(self):
        return self.vehicle.position if self.vehicle else np.array([0, len(self.road.lanes) / 2 * 4])

    def display(self):
        self.road_surface.move_display_window_to(self.display_position())
        self.road.display_road(self.road_surface)
        if self.trajectory:
            self.vehicle.display_trajectory(self.road_surface, self.trajectory)
        self.road.display_traffic(self.road_surface)
        self.screen.blit(self.road_surface, (0, 0))

        if self.smdp:
            self.smdp.display(self.value_surface)
            self.screen.blit(self.value_surface, (0, self.SCREEN_HEIGHT / 2))
        self.clock.tick(self.FPS)
        pygame.display.flip()

        if self.RECORD_VIDEO:
            if not os.path.exists(self.OUT_FOLDER):
                os.mkdir(self.OUT_FOLDER)
            pygame.image.save(self.screen, "{}/highway_{}.bmp".format(self.OUT_FOLDER, self.t))
            if self.vehicle.position[0] > \
                    np.max([o.position[0] for o in self.road.vehicles if o is not self.vehicle]) + 125:
                os.system("ffmpeg -r 60 -i out/highway_%d.bmp -vcodec libx264 -crf 25 out/highway.avi")
                os.system("rm out/*.bmp")
                self.done = True

    def quit(self):
        if self.displayed:
            pygame.quit()


def test():
    road = Road.create_random_road(lanes_count=4, lane_width=4.0, vehicles_count=50, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=MDPVehicle)
    sim.RECORD_VIDEO = True
    while not sim.done:
        sim.process()
    sim.quit()


if __name__ == '__main__':
    test()
