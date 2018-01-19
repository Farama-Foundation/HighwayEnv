from __future__ import division, print_function
import numpy as np
import random
import pygame

from vehicle import Vehicle, ControlledVehicle, MDPVehicle

class Lane(object):
    def __init__(self):
        pass

    def position(self, s):
        pass

    def heading(self, s):
        pass

    def local_coordinates(self, position):
        pass

class StraightLane(Lane):
    STRIPE_SPACING = 5
    STRIPE_LENGTH = 3
    def __init__(self, origin, heading, width, is_road_side=None):
        self.origin = origin
        self.heading = heading
        self.width = width
        self.is_road_side = is_road_side or [False, False]
        self.direction = np.array([np.cos(self.heading), np.sin(self.heading)])
        self.direction_lat = np.array([-self.direction[1], self.direction[0]])

    def position(self, s, lateral):
        return self.origin + s*self.direction + lateral*self.direction_lat

    def heading_at(self, s):
        return self.heading

    def local_coordinates(self, position):
        delta = position - self.origin
        longi = np.dot(delta, self.direction)
        lat = np.dot(delta, self.direction_lat)
        return longi, lat

    def on_lane(self, position):
        longi, lat = self.local_coordinates(position)
        return np.abs(lat) <= self.width

    def display(self, screen):
        s_origin, _ = self.local_coordinates(screen.origin)
        s0 = (int(s_origin)//self.STRIPE_SPACING)*self.STRIPE_SPACING
        ticks = int((screen.get_height()+screen.get_width())/(self.STRIPE_SPACING*screen.SCALING))+1
        for side in range(2):
            self.continuous_line(screen, ticks, s0, side) if self.is_road_side[side] else self.striped_line(screen, ticks, s0, side)

    def continuous_line(self, screen, ticks, s0, side):
        stripe_start = self.position(s0 + 0*self.STRIPE_SPACING, (side-0.5)*self.width)
        stripe_end = self.position(s0 + ticks*self.STRIPE_SPACING + self.STRIPE_LENGTH, (side-0.5)*self.width)
        pygame.draw.line(screen, screen.WHITE,
            (screen.vec2pix(stripe_start)),
            (screen.vec2pix(stripe_end)), 2)

    def striped_line(self, screen, ticks, s0, side):
        for k in range(ticks):
            stripe_start = self.position(s0 + k*self.STRIPE_SPACING, (side-0.5)*self.width)
            stripe_end = self.position(s0 + k*self.STRIPE_SPACING + self.STRIPE_LENGTH, (side-0.5)*self.width)
            pygame.draw.line(screen, screen.WHITE,
                (screen.vec2pix(stripe_start)),
                (screen.vec2pix(stripe_end)), 2)

class SineLane(StraightLane):
    STRIPE_SPACING = 5
    STRIPE_LENGTH = 3
    def __init__(self, origin, heading, width, amplitude, pulsation, is_road_side=None):
        super(SineLane, self).__init__(origin, heading, width, is_road_side)
        self.amplitude = amplitude
        self.pulsation = pulsation

    def position(self, s, lateral):
        return super(SineLane, self).position(s, lateral+self.amplitude*np.sin(self.pulsation*s))

    def heading_at(self, s):
        return super(SineLane, self).heading_at(s)+np.arctan(self.amplitude*self.pulsation*np.cos(self.pulsation*s))

    def local_coordinates(self, position):
        longi, lat = super(SineLane, self).local_coordinates(position)
        return longi, lat-self.amplitude*np.sin(self.pulsation*longi)

    def on_lane(self, position):
        return super(SineLane, self).on_lane(screen)

    def display(self, screen):
        super(SineLane, self).display(screen)

class Road(object):
    """
        The set of vehicles on the road, and its characteristics
    """
    def __init__(self, lanes_count, lane_width, vehicles=[]):
        self.lanes = []
        for l in range(lanes_count):
            origin = np.array([0,l*lane_width])
            heading = 0
            is_road_side = [l==0, l==lanes_count-1]
            self.lanes.append(StraightLane(origin, heading, lane_width, is_road_side))

        self.lane_width = lane_width
        self.vehicles = vehicles


    @classmethod
    def create_random_road(cls, lanes, lane_width, vehicles_count=100):
        r = Road(lanes, lane_width)
        for _ in range(vehicles_count):
            r.vehicles.append(ControlledVehicle.create_random(r))
        return r

    def step(self, dt):
        for vehicle in self.vehicles:
            vehicle.step(dt)

    def get_lane(self, position):
        return self.lanes[self.get_lane_index(position)]

    def get_lane_index(self, position):
        lateral = [abs(l.local_coordinates(position)[1]) for l in self.lanes]
        return np.argmin(lateral)

    def get_lane_coordinates(self, lane, position):
        return self.lanes[lane].local_coordinates(position)

    def move_display_window_to(self, screen, position):
        screen.origin = position - np.array([15, screen.get_height()/(2*screen.SCALING)])

    def display_road(self, screen):
        screen.fill(screen.GREY)
        for l in self.lanes:
            l.display(screen)

    def display_traffic(self, screen):
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
        self.origin = np.array([0, 0])

    def pix(self, length):
        return int(length*self.SCALING)

    def pos2pix(self, x, y):
        return self.pix(x-self.origin[0]), self.pix(y-self.origin[1])

    def vec2pix(self, vec):
        return self.pos2pix(vec[0], vec[1])

def test():
    from simulation import Simulation
    sim = Simulation(lanes_count=1, vehicles_count=0)
    l = SineLane(sim.road.lanes[-1].origin+np.array([0,9]),0,4.0, 3, 6.28/60, [False,False])
    sim.road.lanes.append(l)
    for _ in range(50):
        sim.road.vehicles.append(ControlledVehicle.create_random(sim.road))
    sim.vehicle.position[0] = sim.road.vehicles[-1].position[0]-10

    while not sim.done:
        sim.process()
    sim.quit()

if __name__ == '__main__':
    test()