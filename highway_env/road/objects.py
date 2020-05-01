from abc import ABC
from typing import Sequence

import numpy as np


class RoadObject(ABC):
    """
        Common interface for objects that appear on the road, beside vehicles.
        For now we assume all objects are rectangular.
    """
    LENGTH = 2.0  # Object length [m]
    WIDTH = 2.0  # Object width [m]

    def __init__(self, road, position: Sequence[float], heading: float = 0.):
        """
        :param road: the road instance where the object is placed in
        :param position: cartesian position of object in the surface
        :param heading: the angle from positive direction of horizontal axis
        """
        self.road = road
        self.position = np.array(position, dtype=np.float)
        self.heading = heading
        # store whether object is hit by any vehicle
        self.hit = False

    @classmethod
    def make_on_lane(cls, road, lane_index: tuple, longitudinal: int):
        """
            Create an object on a given lane at a longitudinal position.

        :param road: the road instance where the object is placed in
        :param lane_index: a tuple (origin node, destination node, lane id on the road).
        :param longitudinal: longitudinal position along the lane
        :return: An object with at the specified position
        """
        lane = road.network.get_lane(lane_index)
        return cls(road, lane.position(longitudinal, 0), lane.heading_at(longitudinal))

    # Just added for sake of compatibility, TODO: change usages the way that this make sense.
    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': 0.,
            'vy': 0.,
            'cos_h': np.cos(self.heading),
            'sin_h': np.sin(self.heading),
            'cos_d': 0.,
            'sin_d': 0.
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def __str__(self):
        return f"{self.__class__.__name__} #{id(self) % 1000}: at {self.position}"

    def __repr__(self):
        return self.__str__()


class Obstacle(RoadObject):
    """
        Obstacles on the road.
    """
    pass


class Landmark(RoadObject):
    """
        Landmarks of certain areas on the road that must be reached
    """
    pass
