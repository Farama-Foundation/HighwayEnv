from __future__ import division, print_function
import numpy as np
import random
import pygame

from vehicle import Vehicle, ControlledVehicle, MDPVehicle, IDMVehicle

class Lane(object):
    STRIPE_SPACING = 5
    STRIPE_LENGTH = 3
    STRIPE_WIDTH = 0.3

    def __init__(self):
        raise Exception('Not implemented.')

    def position(self, s):
        raise Exception('Not implemented.')

    def heading(self, s):
        raise Exception('Not implemented.')

    def local_coordinates(self, position):
        raise Exception('Not implemented.')

class StraightLane(Lane):
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

    def display(self, screen, s_bounds=[-np.inf, np.inf]):
        s_origin, _ = self.local_coordinates(screen.origin)
        s0 = (int(s_origin)//self.STRIPE_SPACING)*self.STRIPE_SPACING
        stripes_count = int((screen.get_height()+screen.get_width())/(self.STRIPE_SPACING*screen.SCALING))+1
        for side in range(2):
            if self.is_road_side[side]:
                self.continuous_line(screen, stripes_count, s0, side, s_bounds)
            else:
                self.striped_line(screen, stripes_count, s0, side, s_bounds)

    def continuous_line(self, screen, stripes_count, s0, side, s_bounds):
        stripe_start = min(max(s0 + 0*self.STRIPE_SPACING, s_bounds[0]), s_bounds[1])
        stripe_end = min(max(s0 + stripes_count*self.STRIPE_SPACING + self.STRIPE_LENGTH, s_bounds[0]), s_bounds[1])
        stripe_lat = (side-0.5)*self.width
        pygame.draw.line(screen, screen.WHITE,
            (screen.vec2pix(self.position(stripe_start, stripe_lat))),
            (screen.vec2pix(self.position(stripe_end, stripe_lat))), max(screen.pix(self.STRIPE_WIDTH),1))

    def striped_line(self, screen, stripes_count, s0, side, s_bounds):
        for k in range(stripes_count):
            stripe_start = min(max(s0 + k*self.STRIPE_SPACING, s_bounds[0]), s_bounds[1])
            stripe_end = min(max(s0 + k*self.STRIPE_SPACING + self.STRIPE_LENGTH, s_bounds[0]), s_bounds[1])
            stripe_lat = (side-0.5)*self.width
            pygame.draw.line(screen, screen.WHITE,
                (screen.vec2pix(self.position(stripe_start, stripe_lat))),
                (screen.vec2pix(self.position(stripe_end, stripe_lat))), max(screen.pix(self.STRIPE_WIDTH),1))

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
        self.update_lanes()


    @classmethod
    def create_random_road(cls, lanes, lane_width, vehicles_count=50, vehicles_type=ControlledVehicle):
        r = Road(lanes, lane_width)
        for _ in range(vehicles_count):
            r.vehicles.append(vehicles_type.create_random(r))
        r.update_lanes()
        return r

    def step(self, dt):
        for vehicle in self.vehicles:
            vehicle.step(dt)
        self.update_lanes()

    def update_lanes(self):
        for v in self.vehicles:
            v.lane = self.get_lane(v.position)

    def get_lane(self, position):
        return self.lanes[self.get_lane_index(position)]

    def get_lane_index(self, position):
        lateral = [abs(l.local_coordinates(position)[1]) for l in self.lanes]
        return np.argmin(lateral)

    def get_lane_coordinates(self, lane, position):
        return self.lanes[lane].local_coordinates(position)

    def front_vehicle(self, vehicle):
        lane = vehicle.lane
        if not lane:
            return None
        s = lane.local_coordinates(vehicle.position)[0]
        s_min = None
        v_min = None
        for v in self.vehicles:
            if v is not vehicle and v.lane == lane:
                s_v, _ = lane.local_coordinates(v.position)
                if s < s_v and (s_min is None or s_v < s_min):
                    s_min = s_v
                    v_min = v
        return v_min

    def move_display_window_to(self, screen, position):
        screen.origin = position - np.array([0.3*screen.get_width()/screen.SCALING, screen.get_height()/(2*screen.SCALING)])

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