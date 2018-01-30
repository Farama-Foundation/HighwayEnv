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

    def heading_at(self, s):
        raise Exception('Not implemented.')

    def width_at(self, s):
        raise Exception('Not implemented.')

    def local_coordinates(self, position):
        raise Exception('Not implemented.')

class LineType:
    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2

class StraightLane(Lane):
    def __init__(self, origin, heading, width, line_types=None):
        self.origin = origin
        self.heading = heading
        self.width = width
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.direction = np.array([np.cos(self.heading), np.sin(self.heading)])
        self.direction_lat = np.array([-self.direction[1], self.direction[0]])

    def position(self, s, lateral):
        return self.origin + s*self.direction + lateral*self.direction_lat

    def heading_at(self, s):
        return self.heading

    def width_at(self, s):
        return self.width

    def local_coordinates(self, position):
        delta = position - self.origin
        longi = np.dot(delta, self.direction)
        lat = np.dot(delta, self.direction_lat)
        return longi, lat

    def on_lane(self, position):
        longi, lat = self.local_coordinates(position)
        return np.abs(lat) <= self.width

    def display(self, screen, bounds=[-np.inf, np.inf]):
        stripes_count = int(2*(screen.get_height()+screen.get_width())/(self.STRIPE_SPACING*screen.scaling))
        s_origin, _ = self.local_coordinates(screen.origin)
        s0 = (int(s_origin)//self.STRIPE_SPACING - stripes_count//2)*self.STRIPE_SPACING
        for side in range(2):
            if self.line_types[side] == LineType.STRIPED:
                self.striped_line(screen, stripes_count, s0, side, bounds)
            elif self.line_types[side] == LineType.CONTINUOUS:
                self.continuous_line(screen, stripes_count, s0, side, bounds)

    def striped_line(self, screen, stripes_count, s0, side, bounds):
        starts = s0 + np.arange(stripes_count)*self.STRIPE_SPACING
        ends = s0 + np.arange(stripes_count)*self.STRIPE_SPACING+self.STRIPE_LENGTH
        lat = (side-0.5)*self.width
        self.draw_stripes(screen, starts, ends, lat, bounds)

    def continuous_line(self, screen, stripes_count, s0, side, bounds):
        starts = [s0 + 0*self.STRIPE_SPACING]
        ends = [s0 + stripes_count*self.STRIPE_SPACING + self.STRIPE_LENGTH]
        lat = (side-0.5)*self.width
        self.draw_stripes(screen, starts, ends, lat, bounds)

    def draw_stripes(self, screen, starts, ends, lat, bounds):
        starts = np.minimum(np.maximum(starts, bounds[0]), bounds[1])
        ends = np.minimum(np.maximum(ends, bounds[0]), bounds[1])
        for k in range(len(starts)):
            if abs(starts[k]-ends[k]) > 0.5*self.STRIPE_LENGTH:
                pygame.draw.line(screen, screen.WHITE,
                    (screen.vec2pix(self.position(starts[k], lat))),
                    (screen.vec2pix(self.position(ends[k], lat))), max(screen.pix(self.STRIPE_WIDTH),1))

class SineLane(StraightLane):
    STRIPE_SPACING = 5
    STRIPE_LENGTH = 3
    def __init__(self, origin, heading, width, amplitude, pulsation, line_types=None):
        super(SineLane, self).__init__(origin, heading, width, line_types)
        self.amplitude = amplitude
        self.pulsation = pulsation

    def position(self, s, lateral):
        return super(SineLane, self).position(s, lateral+self.amplitude*np.sin(self.pulsation*s))

    def heading_at(self, s):
        return super(SineLane, self).heading_at(s)+np.arctan(self.amplitude*self.pulsation*np.cos(self.pulsation*s))

    def local_coordinates(self, position):
        longi, lat = super(SineLane, self).local_coordinates(position)
        return longi, lat-self.amplitude*np.sin(self.pulsation*longi)

class LanesConcatenation(Lane):
    def __init__(self, lanes, end_abscissas):
        self.lanes = lanes
        self.end_abscissas = end_abscissas

    def find_segment(self, s):
        segment = 0
        s_segment = s
        for i in range(len(self.end_abscissas)-1):
            if self.end_abscissas[i] > s_segment:
                break
            else:
                segment = i+1
                s_segment -= self.end_abscissas[i]
        return segment, s_segment

    def position(self, s, lateral):
        segment, s_segment = self.find_segment(s)
        return self.lanes[segment].position(s_segment, lateral)

    def heading_at(self, s):
        segment, s_segment = self.find_segment(s)
        return self.lanes[segment].heading_at(s_segment)

    def width_at(self, s):
        segment, s_segment = self.find_segment(s)
        return self.lanes[segment].width_at(s_segment)

    def local_coordinates(self, position):
        ymin = None
        lane = None
        for i in range(len(self.lanes)):
            x,y = self.lanes[i].local_coordinates(position)
            if (x > -self.STRIPE_SPACING or i == 0) and (x < self.end_abscissas[i] or i == len(self.lanes)-1):
                if ymin is None or abs(y) < ymin:
                    ymin = abs(y)
                    lane = i
        x,y = self.lanes[lane].local_coordinates(position)
        x += np.sum(self.end_abscissas[:lane])
        return x,y

    def display(self, screen):
        for i in range(len(self.lanes)):
            bounds = [0, self.end_abscissas[i]] if i>0 else [-np.inf, self.end_abscissas[i]]
            self.lanes[i].display(screen, bounds)


class Road(object):
    """
        The set of vehicles on the road, and its characteristics
    """
    def __init__(self, lanes=[], vehicles=[]):
        self.lanes = lanes
        self.vehicles = vehicles
        self.update_lanes()


    @classmethod
    def create_random_road(cls, lanes_count, lane_width, vehicles_count=50, vehicles_type=ControlledVehicle):
        lanes = []
        for l in range(lanes_count):
            origin = np.array([0,l*lane_width])
            heading = 0
            line_types = [LineType.CONTINUOUS if l==0 else LineType.STRIPED, LineType.CONTINUOUS if l==lanes_count-1 else LineType.NONE]
            lanes.append(StraightLane(origin, heading, lane_width, line_types))
        r = Road(lanes)
        r.add_random_vehicles(vehicles_count, vehicles_type)
        return r

    def add_random_vehicles(self, vehicles_count=50, vehicles_type=ControlledVehicle):
        for _ in range(vehicles_count):
            self.vehicles.append(vehicles_type.create_random(self))
            self.update_lanes()

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

    def neighbour_vehicles(self, vehicle, lane=None):
        lane = lane or vehicle.lane
        if not lane:
            return None, None
        s = lane.local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles:
            if v is not vehicle and v.lane == lane:
                s_v, _ = lane.local_coordinates(v.position)
                if s < s_v and (s_front is None or s_v < s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

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
    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1

    def __init__(self, size, flags, surf):
        super(RoadSurface, self).__init__(size, flags, surf)
        self.origin = np.array([0, 0])
        self.scaling = 10.0
        self.centering_position = 0.5

    def pix(self, length):
        return int(length*self.scaling)

    def pos2pix(self, x, y):
        return self.pix(x-self.origin[0]), self.pix(y-self.origin[1])

    def vec2pix(self, vec):
        return self.pos2pix(vec[0], vec[1])

    def move_display_window_to(self, position):
        self.origin = position - np.array([self.centering_position*self.get_width()/self.scaling, self.get_height()/(2*self.scaling)])

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                self.scaling *= 1/self.SCALING_FACTOR
            if event.key == pygame.K_o:
                self.scaling *= self.SCALING_FACTOR
            if event.key == pygame.K_m:
                self.centering_position -= self.MOVING_FACTOR
            if event.key == pygame.K_k:
                self.centering_position += self.MOVING_FACTOR


def test():
    from simulation import Simulation
    # l = SineLane(sim.road.lanes[-1].origin+np.array([0,9]),0, 4.0, 3, 6.28/60, [False,False])
    ends = [100, 20, np.inf]
    l0 = StraightLane(np.array([0,0]), 0, 4.0, [LineType.CONTINUOUS, LineType.NONE])
    l1 = StraightLane(np.array([0,4]), 0, 4.0, [LineType.STRIPED, LineType.CONTINUOUS])

    lc0 = StraightLane(np.array([0,6.5+4+4]), 0, 4.0, [LineType.CONTINUOUS, LineType.CONTINUOUS])
    lc1 = StraightLane(lc0.position(ends[0],0), -20*3.14159/180, 4.0, [LineType.CONTINUOUS, LineType.CONTINUOUS])
    lc2 = StraightLane(lc1.position(ends[1],0), 0, 4.0, [LineType.NONE, LineType.CONTINUOUS])
    l2 = LanesConcatenation([lc0, lc1, lc2], ends)
    road = Road([l0, l1, l2])
    road.add_random_vehicles(30, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=ControlledVehicle)

    while not sim.done:
        sim.process()
    sim.quit()

if __name__ == '__main__':
    test()