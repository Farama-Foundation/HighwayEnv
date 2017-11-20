from __future__ import division, print_function
import numpy as np
import random
import pygame

from vehicle import Vehicle, ControlledVehicle

class Road(object):
    """
        The set of vehicles on the road, and its characteristics
    """
    STRIPE_SPACING = 5
    STRIPE_LENGTH = 3
    def __init__(self, lanes, lane_width, vehicles=[]):
        self.lanes = lanes
        self.lane_width = lane_width
        self.vehicles = vehicles

    @classmethod
    def create_random_road(cls, lanes, lane_width, vehicles_count=100):
        r = Road(lanes, lane_width)
        for _ in range(vehicles_count):
            r.vehicles.append(r.random_controlled_vehicle())
        return r

    @classmethod
    def create_obstacles_road(cls, lanes, lane_width, rows=4):
        r = Road(lanes, lane_width)
        for d in range(1,rows+1):
            for l in range(r.lanes-1):
                v = Vehicle([5*d*r.STRIPE_SPACING, r.get_lateral_position(l + d%2)])
                r.vehicles.append(ControlledVehicle.create_from(v, r))
        return r


    def step(self, dt):
        for vehicle in self.vehicles:
            vehicle.step(dt)

    def get_lane(self, position):
        return int(np.floor(position[1]/self.lane_width))

    def get_lateral_position(self, lane):
        return (lane+0.5)*self.lane_width

    def random_vehicle(self, velocity=None, ego=False):
        l = random.randint(0,self.lanes-1)
        xmin = np.min([v.position[0] for v in self.vehicles]) if len(self.vehicles) else 0
        v = Vehicle([xmin-2*self.STRIPE_SPACING, self.get_lateral_position(l)], 0, velocity, ego)
        return v

    def random_controlled_vehicle(self, velocity=None, ego=False):
        return ControlledVehicle.create_from(self.random_vehicle(velocity, ego), self)

    def display(self, screen):
        screen.fill(screen.GREY)
        if len(self.vehicles):
            screen.origin = self.vehicles[-1].position-np.array([3*self.STRIPE_SPACING, screen.get_height()/(2*screen.SCALING)])

        # Draw tracks
        x0 = (int(screen.origin[0])//self.STRIPE_SPACING)*self.STRIPE_SPACING
        ticks = int(screen.get_width()/(self.STRIPE_SPACING*screen.SCALING))+1
        # Outer
        pygame.draw.line(screen, screen.WHITE,
            (screen.pos2pix(x0 + 0*self.STRIPE_SPACING, 0*self.lane_width)),
            (screen.pos2pix(x0 + ticks*self.STRIPE_SPACING + self.STRIPE_LENGTH, 0*self.lane_width)), 2)
        pygame.draw.line(screen, screen.WHITE,
            (screen.pos2pix(x0 + 0*self.STRIPE_SPACING, self.lanes*self.lane_width)),
            (screen.pos2pix(x0 + ticks*self.STRIPE_SPACING + self.STRIPE_LENGTH, self.lanes*self.lane_width)), 2)
        # Inner
        for l in range(1,self.lanes):
            for k in range(ticks):
                pygame.draw.line(screen, screen.WHITE,
                    (screen.pos2pix(x0 + k*self.STRIPE_SPACING, l*self.lane_width)),
                    (screen.pos2pix(x0 + k*self.STRIPE_SPACING + self.STRIPE_LENGTH, l*self.lane_width)), 2)

        for v in self.vehicles:
            v.display(screen)

    def __repr__(self):
        return self.vehicles.__repr__()


class RoadSurface(pygame.Surface):
    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)
    SCALING = 10.0

    def __init__(self, size, flags, surf):
        super(RoadSurface, self).__init__(size, flags, surf)
        self.origin = [0, 0]

    def pix(self, length):
        return int(length*self.SCALING)

    def pos2pix(self, x, y):
        return self.pix(x-self.origin[0]), self.pix(y-self.origin[1])



def test():
    r = Road.create_random_road(4, 4.0, vehicles_count=3)
    print(r)
    for _ in range(10):
        r.step(0.1)
    print(r)


if __name__ == '__main__':
    test()