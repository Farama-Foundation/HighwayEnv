from __future__ import division, print_function
import numpy as np
import pandas as pd

from highway.logger import Loggable
from highway.road.lane import LineType, StraightLane
from highway.vehicle.control import ControlledVehicle
from highway.vehicle.dynamics import Obstacle


class Road(Loggable):
    """
        The set of vehicles on the road, and its characteristics
    """

    def __init__(self, lanes=None, vehicles=None):
        self.lanes = lanes or []
        self.vehicles = vehicles or []

    @classmethod
    def create_random_road(cls, lanes_count, lane_width, vehicles_count=50, vehicles_type=ControlledVehicle):
        lanes = []
        for lane in range(lanes_count):
            origin = np.array([0, lane * lane_width])
            heading = 0
            line_types = [LineType.CONTINUOUS if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS if lane == lanes_count - 1 else LineType.NONE]
            lanes.append(StraightLane(origin, heading, lane_width, line_types))
        r = Road(lanes)
        r.add_random_vehicles(vehicles_count, vehicles_type)
        return r

    def add_random_vehicles(self, vehicles_count=50, vehicles_type=ControlledVehicle):
        for _ in range(vehicles_count):
            self.vehicles.append(vehicles_type.create_random(self))

    def act(self):
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt):
        for vehicle in self.vehicles:
            vehicle.step(dt)
            for other in self.vehicles:
                vehicle.check_collision(other)

    def get_lane(self, position):
        return self.lanes[self.get_lane_index(position)]

    def get_lane_index(self, position):
        lateral = [abs(l.local_coordinates(position)[1]) for l in self.lanes]
        return int(np.argmin(lateral))

    def neighbour_vehicles(self, vehicle, lane=None):
        lane = lane or vehicle.lane
        if not lane:
            return None, None
        s = lane.local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles:
            if v is not vehicle and v.lane == lane:
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    def dump(self):
        for v in self.vehicles:
            if not isinstance(v, Obstacle):
                v.dump()

    def get_log(self):
        return pd.concat([v.get_log() for v in self.vehicles])

    def __repr__(self):
        return self.vehicles.__repr__()
