import numpy as np
from scipy import interpolate
from typing import List, Tuple


class LinearSpline2D:
    """
    Piece-wise linear curve fitted to a list of points.
    """

    PARAM_CURVE_SAMPLE_DISTANCE: int = 1  # curve samples are placed 1m apart

    def __init__(self, points: List[Tuple[float, float]]):
        x_values = np.array([pt[0] for pt in points])
        y_values = np.array([pt[1] for pt in points])
        x_values_diff = np.diff(x_values)
        x_values_diff = np.hstack((x_values_diff, x_values_diff[-1]))
        y_values_diff = np.diff(y_values)
        y_values_diff = np.hstack((y_values_diff, y_values_diff[-1]))
        arc_length_cumulated = np.hstack(
            (0, np.cumsum(np.sqrt(x_values_diff[:-1] ** 2 + y_values_diff[:-1] ** 2)))
        )
        self.length = arc_length_cumulated[-1]
        self.x_curve = interpolate.interp1d(
            arc_length_cumulated, x_values, fill_value="extrapolate"
        )
        self.y_curve = interpolate.interp1d(
            arc_length_cumulated, y_values, fill_value="extrapolate"
        )
        self.dx_curve = interpolate.interp1d(
            arc_length_cumulated, x_values_diff, fill_value="extrapolate"
        )
        self.dy_curve = interpolate.interp1d(
            arc_length_cumulated, y_values_diff, fill_value="extrapolate"
        )

        (self.s_samples, self.poses) = self.sample_curve(
            self.x_curve, self.y_curve, self.length, self.PARAM_CURVE_SAMPLE_DISTANCE
        )

    def __call__(self, lon: float) -> Tuple[float, float]:
        return self.x_curve(lon), self.y_curve(lon)

    def get_dx_dy(self, lon: float) -> Tuple[float, float]:
        idx_pose = self._get_idx_segment_for_lon(lon)
        pose = self.poses[idx_pose]
        return pose.normal

    def cartesian_to_frenet(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform the point in Cartesian coordinates into Frenet coordinates of the curve
        """

        pose = self.poses[-1]
        projection = pose.project_onto_normal(position)
        if projection >= 0:
            lon = self.s_samples[-1] + projection
            lat = pose.project_onto_orthonormal(position)
            return lon, lat

        for idx in list(range(len(self.s_samples) - 1))[::-1]:
            pose = self.poses[idx]
            projection = pose.project_onto_normal(position)
            if projection >= 0:
                if projection < pose.distance_to_origin(position):
                    lon = self.s_samples[idx] + projection
                    lat = pose.project_onto_orthonormal(position)
                    return lon, lat
                else:
                    ValueError("No valid projection could be found")
        pose = self.poses[0]
        lon = pose.project_onto_normal(position)
        lat = pose.project_onto_orthonormal(position)
        return lon, lat

    def frenet_to_cartesian(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        Convert the point from Frenet coordinates of the curve into Cartesian coordinates
        """
        idx_segment = self._get_idx_segment_for_lon(lon)
        s = lon - self.s_samples[idx_segment]
        pose = self.poses[idx_segment]
        point = pose.position + s * pose.normal
        point += lat * pose.orthonormal
        return point

    def _get_idx_segment_for_lon(self, lon: float) -> int:
        """
        Returns the index of the curve pose that corresponds to the longitudinal coordinate
        """
        idx_smaller = np.argwhere(lon < self.s_samples)
        if len(idx_smaller) == 0:
            return len(self.s_samples) - 1
        if idx_smaller[0] == 0:
            return 0
        return int(idx_smaller[0]) - 1

    @staticmethod
    def sample_curve(x_curve, y_curve, length: float, CURVE_SAMPLE_DISTANCE=1):
        """
        Create samples of the curve that are CURVE_SAMPLE_DISTANCE apart. These samples are used for Frenet to Cartesian
        conversion and vice versa
        """
        num_samples = np.floor(length / CURVE_SAMPLE_DISTANCE)
        s_values = np.hstack(
            (CURVE_SAMPLE_DISTANCE * np.arange(0, int(num_samples) + 1), length)
        )
        x_values = x_curve(s_values)
        y_values = y_curve(s_values)
        dx_values = np.diff(x_values)
        dx_values = np.hstack((dx_values, dx_values[-1]))
        dy_values = np.diff(y_values)
        dy_values = np.hstack((dy_values, dy_values[-1]))

        poses = [
            CurvePose(x, y, dx, dy)
            for x, y, dx, dy in zip(x_values, y_values, dx_values, dy_values)
        ]

        return s_values, poses


class CurvePose:
    """
    Sample pose on a curve that is used for Frenet to Cartesian conversion
    """

    def __init__(self, x: float, y: float, dx: float, dy: float):
        self.length = np.sqrt(dx**2 + dy**2)
        self.position = np.array([x, y]).flatten()
        self.normal = np.array([dx, dy]).flatten() / self.length
        self.orthonormal = np.array([-self.normal[1], self.normal[0]]).flatten()

    def distance_to_origin(self, point: Tuple[float, float]) -> float:
        """
        Compute the distance between the point [x, y] and the pose origin
        """
        return np.sqrt(np.sum((self.position - point) ** 2))

    def project_onto_normal(self, point: Tuple[float, float]) -> float:
        """
        Compute the longitudinal distance from pose origin to point by projecting the point onto the normal vector of the pose
        """
        return self.normal.dot(point - self.position)

    def project_onto_orthonormal(self, point: Tuple[float, float]) -> float:
        """
        Compute the lateral distance from pose origin to point by projecting the point onto the orthonormal vector of the pose
        """
        return self.orthonormal.dot(point - self.position)
