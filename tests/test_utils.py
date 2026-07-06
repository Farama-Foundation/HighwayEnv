import numpy as np
import pytest

from highway_env.utils import (
    confidence_ellipsoid,
    confidence_polytope,
    distance_to_circle,
    distance_to_rect,
    is_consistent_dataset,
    is_valid_observation,
    near_split,
    rotated_rectangles_intersect,
    solve_trinom,
)


def test_rotated_rectangles_intersect():
    assert rotated_rectangles_intersect(
        ([12.86076812, 28.60182391], 5.0, 2.0, -0.4675779906495494),
        ([9.67753944, 28.90585412], 5.0, 2.0, -0.3417019364473201),
    )
    assert rotated_rectangles_intersect(([0, 0], 2, 1, 0), ([0, 1], 2, 1, 0))
    assert not rotated_rectangles_intersect(([0, 0], 2, 1, 0), ([0, 2.1], 2, 1, 0))
    assert not rotated_rectangles_intersect(([0, 0], 2, 1, 0), ([1, 1.1], 2, 1, 0))
    assert rotated_rectangles_intersect(([0, 0], 2, 1, np.pi / 4), ([1, 1.1], 2, 1, 0))


@pytest.fixture
def linear_regression_data():
    return {
        "features": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        "outputs": [1.0, 2.0, 3.0],
    }


def test_confidence_ellipsoid(linear_regression_data):
    theta, gramian, beta = confidence_ellipsoid(linear_regression_data)
    assert theta.shape == (2,)
    assert gramian.shape == (2, 2)
    assert beta > 0


def test_confidence_polytope(linear_regression_data):
    parameter_box = np.array([[-10.0, -10.0], [10.0, 10.0]])
    theta, d_theta, gramian, beta = confidence_polytope(
        linear_regression_data, parameter_box
    )
    assert theta.shape == (2,)
    assert d_theta.shape[0] == 2 ** theta.shape[0]
    assert gramian.shape == (2, 2)
    assert beta > 0
    assert np.all(theta >= parameter_box[0])
    assert np.all(theta <= parameter_box[1])


def test_is_valid_observation(linear_regression_data):
    parameter_box = np.array([[-10.0, -10.0], [10.0, 10.0]])
    theta, _, gramian, beta = confidence_polytope(linear_regression_data, parameter_box)
    phi = np.array([1.0, 0.0])[..., np.newaxis]
    y = np.array([theta @ phi])[..., np.newaxis]
    assert is_valid_observation(y, phi, theta, gramian, beta)
    assert not is_valid_observation(y + 100, phi, theta, gramian, beta)


def test_is_consistent_dataset_single_point():
    data = {"features": [[1.0, 0.0]], "outputs": [1.0]}
    assert is_consistent_dataset(data)


def test_is_consistent_dataset_multiple_points(linear_regression_data):
    parameter_box = np.array([[-10.0, -10.0], [10.0, 10.0]])
    assert is_consistent_dataset(linear_regression_data, parameter_box=parameter_box)


def test_near_split_num_bins():
    assert near_split(10, num_bins=3) == [4, 3, 3]
    assert sum(near_split(10, num_bins=3)) == 10


def test_near_split_size_bins():
    assert near_split(10, size_bins=4) == [4, 3, 3]
    assert sum(near_split(10, size_bins=4)) == 10


def test_distance_to_circle_hit():
    center = np.array([[-2.32503077], [-0.21879166]])
    direction = np.array([[-1.24591095], [-0.73226735]])
    distance = distance_to_circle(center, 1.0, direction)
    assert float(np.asarray(distance).item()) == pytest.approx(1.36356508, rel=1e-5)


def test_distance_to_circle_inside():
    center = np.array([[0.0], [0.0]])
    direction = np.array([[1.0], [0.0]])
    assert distance_to_circle(center, 1.0, direction) == 0


def test_distance_to_circle_miss():
    center = np.array([[5.0], [5.0]])
    direction = np.array([[1.0], [0.0]])
    assert distance_to_circle(center, 1.0, direction) == np.inf


def test_distance_to_rect_intersection():
    rect = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 1.0]),
    ]
    line = (np.array([-1.0, 0.5]), np.array([2.0, 0.5]))
    assert distance_to_rect(line, rect) == pytest.approx(1.0)


def test_distance_to_rect_no_intersection():
    rect = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 1.0]),
    ]
    line = (np.array([-1.0, 2.0]), np.array([2.0, 2.0]))
    assert distance_to_rect(line, rect) == np.inf


def test_solve_trinom():
    assert solve_trinom(1.0, -3.0, 2.0) == pytest.approx((1.0, 2.0))
    assert solve_trinom(1.0, 0.0, 1.0) == (None, None)
