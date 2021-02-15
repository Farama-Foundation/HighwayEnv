from abc import ABC
from typing import Sequence, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from highway_env.road.lane import AbstractLane
    from highway_env.road.road import Road

LaneIndex = Tuple[str, str, int]


class RoadObject(ABC):

    """
    Common interface for objects that appear on the road.

    For now we assume all objects are rectangular.
    """

    LENGTH: float = 2  # Object length [m]
    WIDTH: float = 2  # Object width [m]

    def __init__(self, road: 'Road', position: Sequence[float], heading: float = 0, speed: float = 0):
        """
        :param road: the road instance where the object is placed in
        :param position: cartesian position of object in the surface
        :param heading: the angle from positive direction of horizontal axis
        :param speed: cartesian speed of object in the surface
        """
        self.road = road
        self.position = np.array(position, dtype=np.float)
        self.heading = heading
        self.speed = speed
        self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading) if self.road else np.nan
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None

    @classmethod
    def make_on_lane(cls, road: 'Road', lane_index: LaneIndex, longitudinal: float):
        """
        Create an object on a given lane at a longitudinal position.

        :param road: the road instance where the object is placed in
        :param lane_index: a tuple (origin node, destination node, lane id on the road).
        :param longitudinal: longitudinal position along the lane
        :return: An object with at the specified position
        """
        lane = road.network.get_lane(lane_index)
        return cls(road, position=lane.position(longitudinal, 0), heading=lane.heading_at(longitudinal))

    # Just added for sake of compatibility
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

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction

    def polygon(self) -> np.ndarray:
        points = np.array([
            [-self.LENGTH / 2, -self.WIDTH / 2],
            [-self.LENGTH / 2, +self.WIDTH / 2],
            [+self.LENGTH / 2, +self.WIDTH / 2],
            [+self.LENGTH / 2, -self.WIDTH / 2],
        ]).T
        c, s = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([
            [c, -s],
            [s, c]
        ])
        points = (rotation @ points).T + np.tile(self.position, (4, 1))
        return np.vstack([points, points[0:1]])

    def lane_distance_to(self, other: 'RoadObject', lane: 'AbstractLane' = None) -> float:
        """
        Compute the signed distance to another object along a lane.

        :param other: the other object
        :param lane: a lane
        :return: the distance to the other other [m]
        """
        if not other:
            return np.nan
        if not lane:
            lane = self.lane
        return lane.local_coordinates(other.position)[0] - lane.local_coordinates(self.position)[0]

    @property
    def on_road(self) -> bool:
        """ Is the object on its current lane, or off-road? """
        return self.lane.on_lane(self.position)

    def front_distance_to(self, other: "RoadObject") -> float:
        return self.direction.dot(other.position - self.position)

    def __str__(self):
        return f"{self.__class__.__name__} #{id(self) % 1000}: at {self.position}"

    def __repr__(self):
        return self.__str__()


class Obstacle(RoadObject):

    """Obstacles on the road."""

    def __init__(self, road, position: Sequence[float], heading: float = 0, speed: float = 0):
        super().__init__(road, position, heading, speed)
        # store whether object is hit by any vehicle
        self.hit = False


class Landmark(RoadObject):

    """Landmarks of certain areas on the road that must be reached."""

    def __init__(self, road, position: Sequence[float], heading: float = 0, speed: float = 0):
        super().__init__(road, position, heading, speed)
        # store whether object is hit by any vehicle
        self.hit = False

