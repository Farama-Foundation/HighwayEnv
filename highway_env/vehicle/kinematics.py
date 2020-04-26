import copy
import numpy as np
import pandas as pd
from collections import deque

from highway_env import utils
from highway_env.logger import Loggable


class Vehicle(Loggable):
    """
        A moving vehicle on a road, and its kinematics.

        The vehicle is represented by a dynamical system: a modified bicycle model.
        It's state is propagated depending on its steering and acceleration actions.
    """
    COLLISIONS_ENABLED = True
    """ Enable collision detection between vehicles """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_VELOCITIES = [23, 25]
    """ Range for random initial velocities [m/s] """
    MAX_VELOCITY = 40.
    """ Maximum reachable velocity [m/s] """

    def __init__(self, road, position, heading=0, velocity=0):
        self.road = road
        self.position = np.array(position).astype('float')
        self.heading = heading
        self.velocity = velocity
        self.lane_index = self.road.network.get_closest_lane_index(self.position) if self.road else np.nan
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.crashed_with_obstacle = False
        self.log = []
        self.history = deque(maxlen=30)

    @classmethod
    def make_on_lane(cls, road, lane_index, longitudinal, velocity=0):
        """
            Create a vehicle on a given lane at a longitudinal position.

        :param road: the road where the vehicle is driving
        :param lane_index: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param velocity: initial velocity in [m/s]
        :return: A vehicle with at the specified position
        """
        lane = road.network.get_lane(lane_index)
        if velocity is None:
            velocity = lane.speed_limit
        return cls(road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), velocity)

    @classmethod
    def create_random(cls, road, velocity=None, spacing=1):
        """
            Create a random vehicle on the road.

            The lane and /or velocity are chosen randomly, while longitudinal position is chosen behind the last
            vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or velocity
        """
        if velocity is None:
            velocity = road.np_random.uniform(Vehicle.DEFAULT_VELOCITIES[0], Vehicle.DEFAULT_VELOCITIES[1])
        default_spacing = 1.5*velocity
        _from = road.np_random.choice(list(road.network.graph.keys()))
        _to = road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = road.np_random.choice(len(road.network.graph[_from][_to]))
        offset = spacing * default_spacing * np.exp(-5 / 30 * len(road.network.graph[_from][_to]))
        # x0 = np.max([v.position[0] for v in road.vehicles]) if len(road.vehicles) else 3*offset
        x0 = len(road.vehicles) * 30 if len(road.vehicles) else -10
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road,
                road.network.get_lane((_from, _to, _id)).position(x0, 0),
                road.network.get_lane((_from, _to, _id)).heading_at(x0),
                velocity)
        return v

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.velocity)
        return v

    def act(self, action=None):
        """
            Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt):
        """
            Propagate the vehicle state given its actions.

            Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
            If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
            The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions()
        v = self.velocity * np.array([np.cos(self.heading + self.action['steering']),
                                      np.sin(self.heading + self.action['steering'])])
        self.position += v * dt
        self.heading += self.velocity * np.sin(self.action['steering']) / (self.LENGTH / 2) * dt
        self.velocity += self.action['acceleration'] * dt
        self.on_state_update()

    def clip_actions(self):
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.velocity
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        if self.velocity > self.MAX_VELOCITY:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0*(self.MAX_VELOCITY - self.velocity))
        elif self.velocity < -self.MAX_VELOCITY:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0*(self.MAX_VELOCITY - self.velocity))

    def on_state_update(self):
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def lane_distance_to(self, vehicle, lane=None):
        """
            Compute the signed distance to another vehicle along a lane.

        :param vehicle: the other vehicle
        :param lane: a lane
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        if not lane:
            lane = self.lane
        return lane.local_coordinates(vehicle.position)[0] - lane.local_coordinates(self.position)[0]

    def check_collision(self, other):
        """
            Check for collision with another vehicle.

        :param other: the other vehicle
        """
        if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED or self.crashed or other is self:
            return

        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return

        # Accurate rectangular check
        if utils.rotated_rectangles_intersect((self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),
                                              (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading)):
            self.velocity = other.velocity = min([self.velocity, other.velocity], key=abs)
            if isinstance(other, Obstacle):
                self.crashed_with_obstacle = other.crashed_with_obstacle = True
            elif isinstance(other, Vehicle):
                self.crashed = other.crashed = True

    @property
    def direction(self):
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def destination(self):
        if getattr(self, "route", None):
            last_lane = self.road.network.get_lane(self.route[-1])
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self):
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    def front_distance_to(self, other):
        return self.direction.dot(other.position - self.position)

    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity * self.direction[0],
            'vy': self.velocity * self.direction[1],
            'cos_h': self.direction[0],
            'sin_h': self.direction[1],
            'cos_d': self.destination_direction[0],
            'sin_d': self.destination_direction[1]
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def dump(self):
        """
            Update the internal log of the vehicle, containing:
                - its kinematics;
                - some metrics relative to its neighbour vehicles.
        """
        data = {
            'x': self.position[0],
            'y': self.position[1],
            'psi': self.heading,
            'vx': self.velocity * np.cos(self.heading),
            'vy': self.velocity * np.sin(self.heading),
            'v': self.velocity,
            'acceleration': self.action['acceleration'],
            'steering': self.action['steering']}

        if self.road:
            for lane_index in self.road.network.side_lanes(self.lane_index):
                lane_coords = self.road.network.get_lane(lane_index).local_coordinates(self.position)
                data.update({
                    'dy_lane_{}'.format(lane_index): lane_coords[1],
                    'psi_lane_{}'.format(lane_index): self.road.network.get_lane(lane_index).heading_at(lane_coords[0])
                })
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
            if front_vehicle:
                data.update({
                    'front_v': front_vehicle.velocity,
                    'front_distance': self.lane_distance_to(front_vehicle)
                })
            if rear_vehicle:
                data.update({
                    'rear_v': rear_vehicle.velocity,
                    'rear_distance': rear_vehicle.lane_distance_to(self)
                })

        self.log.append(data)

    def get_log(self):
        """
            Cast the internal log as a DataFrame.

        :return: the DataFrame of the Vehicle's log.
        """
        return pd.DataFrame(self.log)

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()


class Obstacle(Vehicle):
    """
        A motionless obstacle at a given position.
    """

    def __init__(self, road, position, heading=0):
        super(Obstacle, self).__init__(road, position, velocity=0, heading=heading)
        self.target_velocity = 0
        self.LENGTH = self.WIDTH
