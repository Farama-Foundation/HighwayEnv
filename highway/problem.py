from __future__ import division, print_function
import numpy as np
import random
import pygame

BLACK = (0, 0, 0)
GREY = (100, 100, 100)
GREEN = (50, 200, 0)
WHITE = (255, 255, 255)
SCALING = 10.0

origin = [0, 0]

def pix(length):
    return int(length*SCALING)

def pos2pix(x, y):
    global origin
    return pix(x-origin[0]), pix(y-origin[1])

class Vehicle:
    LENGTH = 5.0
    WIDTH = 2.0

    def __init__(self, position, heading, velocity):
        self.position = position
        self.heading = heading
        self.velocity = velocity
        self.action = {'steering':0, 'acceleration':0}
        self.color = GREEN

    def step(self, dt):
        v = self.velocity*np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v*dt
        self.heading += self.velocity*self.action['steering']*dt
        self.velocity += self.action['acceleration']*dt

    def display(self, screen):
        s = pygame.Surface((pix(self.LENGTH), pix(self.WIDTH)), pygame.SRCALPHA)   # per-pixel alpha
        s.fill(GREEN)
        pygame.draw.rect(s, BLACK, (0,0,pix(self.LENGTH),pix(self.WIDTH)), 1)
        s = s.convert_alpha()
        sr = pygame.transform.rotate(s, -self.heading*180/np.pi)
        screen.blit(sr, (pos2pix(self.position[0]-self.LENGTH/2, self.position[1]-self.WIDTH/2)))

    def __str__(self):
        return "{}, {}, {}".format(self.position, self.heading, self.velocity)

class Road:
    STRIPE_SPACING = 5
    STRIPE_LENGTH = 3
    def __init__(self, lanes, lane_width, vehicles=None):
        self.lanes = lanes
        self.lane_width = lane_width
        self.vehicles = vehicles

    def step(self, dt):
        for vehicle in self.vehicles:
            vehicle.step(dt)

    def get_lane(position):
        return int(np.floor(position[1]/self.lane_width))

    def display(self, screen):
        screen.fill(GREY)
        global origin
        if len(self.vehicles):
            origin = self.vehicles[-1].position-np.array([3, screen.get_height()/(2*SCALING)])

        # Draw tracks
        x0 = (int(origin[0])//self.STRIPE_SPACING)*self.STRIPE_SPACING
        ticks = int(screen.get_width()/(self.STRIPE_SPACING*SCALING))+1
        pygame.draw.line(screen, WHITE,
            (pos2pix(x0 + 0*self.STRIPE_SPACING, 0*self.lane_width)),
            (pos2pix(x0 + ticks*self.STRIPE_SPACING + self.STRIPE_LENGTH, 0*self.lane_width)), 2)
        pygame.draw.line(screen, WHITE,
            (pos2pix(x0 + 0*self.STRIPE_SPACING, self.lanes*self.lane_width)),
            (pos2pix(x0 + ticks*self.STRIPE_SPACING + self.STRIPE_LENGTH, self.lanes*self.lane_width)), 2)
        for l in range(1,self.lanes):
            for k in range(ticks):
                pygame.draw.line(screen, WHITE,
                    (pos2pix(x0 + k*self.STRIPE_SPACING, l*self.lane_width)),
                    (pos2pix(x0 + k*self.STRIPE_SPACING + self.STRIPE_LENGTH, l*self.lane_width)), 2)

        for v in self.vehicles:
            v.display(screen)

    def __str__(self):
        s = ""
        for v in self.vehicles:
            s = s + str(v) + '\n'
        return s

def test():
    v = Vehicle(np.array([0, 2.0]), 0, 10)
    r = Road(3, 4.0, [v])

    for _ in range(10):
        r.step()
    print(v)

if __name__ == '__main__':
    test()