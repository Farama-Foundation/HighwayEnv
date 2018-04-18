from __future__ import division, print_function
import numpy as np
import pandas as pd

from highway_env.logger import Loggable
from highway_env.road.lane import LineType, StraightLane
from highway_env.vehicle.control import ControlledVehicle
from highway_env.vehicle.dynamics import Obstacle


class Road(Loggable):
    """
        A road is a set of lanes, and a set of vehicles driving on these lanes
    """

    def __init__(self, lanes=None, vehicles=None):
        """
            New road.

        :param lanes: the lanes composing the road
        :param vehicles: the vehicles driving on the road
        """
        self.lanes = lanes or []
        self.vehicles = vehicles or []

    @classmethod
    def create_random_road(cls,
                           lanes_count,
                           lane_width=4,
                           vehicles_count=50,
                           vehicles_type=ControlledVehicle,
                           np_random=None):
        """
            Create a road composed of straight adjacent lanes with randomly located vehicles on it.

        :param lanes_count: number of lanes
        :param lane_width: lanes width [m]
        :param vehicles_count: number of vehicles on the road
        :param vehicles_type: type of vehicles on the road
        :param np.random.RandomState np_random: a random number generator
        :return: the created road
        """
        lanes = []
        for lane in range(lanes_count):
            origin = np.array([0, lane * lane_width])
            heading = 0
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes_count - 1 else LineType.NONE]
            lanes.append(StraightLane(origin, heading, lane_width, line_types))
        r = Road(lanes)
        r.add_random_vehicles(vehicles_count, vehicles_type, np_random)
        return r

    def add_random_vehicles(self, vehicles_count=50, vehicles_type=ControlledVehicle, np_random=None):
        """
            Create some new random vehicles of a given type, and add them on the road.

        :param vehicles_count: number of vehicles to create
        :param vehicles_type: type of vehicles to create
        :param np.random.RandomState np_random: a random number generator
        """
        for _ in range(vehicles_count):
            self.vehicles.append(vehicles_type.create_random(self, np_random=np_random))

    def act(self):
        """
            Decide the actions of each entity on the road.
        """
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt):
        """
            Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        for vehicle in self.vehicles:
            vehicle.step(dt)
            for other in self.vehicles:
                vehicle.check_collision(other)

    def get_lane(self, position):
        """
            Get the lane closest to a world position.

        :param position: a world position [m]
        :return: the closest lane
        """
        return self.lanes[self.get_lane_index(position)]

    def get_lane_index(self, position):
        """
            Get the index of the lane closest to a world position.

        :param position: a world position [m]
        :return: the index of the closest lane
        """
        lateral = [abs(l.local_coordinates(position)[1]) for l in self.lanes]
        return int(np.argmin(lateral))

    def neighbour_vehicles(self, vehicle, lane=None):
        """
            Find the preceding and following vehicles of a given vehicle.
        :param vehicle: the vehicle whose neighbours must be found
        :param lane: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
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
        """
            Dump the data of all entities on the road
        """
        for v in self.vehicles:
            if not isinstance(v, Obstacle):
                v.dump()

    def get_log(self):
        """
            Concatenate the logs of all entities on the road.
        :return: the concatenated log.
        """
        return pd.concat([v.get_log() for v in self.vehicles])

    def __repr__(self):
        return self.vehicles.__repr__()
