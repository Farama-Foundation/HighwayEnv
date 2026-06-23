import numpy as np
import pytest
from scipy import interpolate

from highway_env.road.spline import LinearSpline2D, numpy_interp1d


@pytest.fixture
def simple_data():
    """Five-point dataset with non-uniform spacing."""
    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    y = np.array([2.0, 3.0, 5.0, -1.0, 7.0])
    return x, y


@pytest.fixture
def two_point_data():
    """Minimal single-segment dataset."""
    x = np.array([1.0, 4.0])
    y = np.array([10.0, 25.0])
    return x, y


def _scipy_ref(x, y):
    return interpolate.interp1d(x, y, fill_value="extrapolate")


class TestRepresentative:
    """Representative cases – interior interpolation."""

    def test_exact_knots(self, simple_data):
        x, y = simple_data
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(x), f_sp(x))

    def test_midpoints(self, simple_data):
        x, y = simple_data
        mids = (x[:-1] + x[1:]) / 2
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(mids), f_sp(mids))

    def test_dense_interior(self, simple_data):
        x, y = simple_data
        query = np.linspace(x[0], x[-1], 200)
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(query), f_sp(query))

    def test_scalar_input(self, simple_data):
        x, y = simple_data
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        for q in [0.5, 2.0, 8.0]:
            assert isinstance(f_np(q), float)
            np.testing.assert_allclose(f_np(q), float(f_sp(q)))

    def test_single_element_array(self, simple_data):
        x, y = simple_data
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        q = np.array([5.0])
        np.testing.assert_allclose(f_np(q), f_sp(q))


class TestBorderline:
    """Borderline cases – at / near boundaries."""

    def test_first_knot(self, simple_data):
        x, y = simple_data
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(x[0]), f_sp(x[0]))

    def test_last_knot(self, simple_data):
        x, y = simple_data
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(x[-1]), f_sp(x[-1]))

    def test_epsilon_inside_left(self, simple_data):
        x, y = simple_data
        q = x[0] + 1e-12
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(q), f_sp(q), atol=1e-10)

    def test_epsilon_outside_left(self, simple_data):
        x, y = simple_data
        q = x[0] - 1e-12
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(q), f_sp(q), atol=1e-10)

    def test_epsilon_inside_right(self, simple_data):
        x, y = simple_data
        q = x[-1] - 1e-12
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(q), f_sp(q), atol=1e-10)

    def test_epsilon_outside_right(self, simple_data):
        x, y = simple_data
        q = x[-1] + 1e-12
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(q), f_sp(q), atol=1e-10)


class TestExtreme:
    """Extreme cases."""

    def test_large_extrapolation_left(self, simple_data):
        x, y = simple_data
        q = x[0] - 1000.0
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(q), f_sp(q))

    def test_large_extrapolation_right(self, simple_data):
        x, y = simple_data
        q = x[-1] + 1000.0
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(q), f_sp(q))

    def test_two_point_interpolation(self, two_point_data):
        x, y = two_point_data
        query = np.linspace(x[0], x[-1], 50)
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(query), f_sp(query))

    def test_two_point_extrapolation(self, two_point_data):
        x, y = two_point_data
        query = np.array([x[0] - 10.0, x[-1] + 10.0])
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(query), f_sp(query))

    def test_large_array(self):
        rng = np.random.default_rng(42)
        x = np.sort(rng.uniform(0, 1000, size=5000))
        y = rng.standard_normal(5000)
        query = np.linspace(-10, 1010, 2000)
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(query), f_sp(query), atol=1e-10)

    def test_negative_x_values(self):
        x = np.array([-10.0, -5.0, -1.0, 0.0, 3.0])
        y = np.array([1.0, 4.0, 2.0, 0.0, 6.0])
        query = np.array([-15.0, -7.5, -3.0, 1.5, 5.0])
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(query), f_sp(query))

    def test_mixed_extrapolation_array(self, simple_data):
        """Query array that includes left-extrapolated, interior, and right-extrapolated points."""
        x, y = simple_data
        query = np.array([-5.0, 0.0, 2.0, 5.0, 10.0, 20.0])
        f_np = numpy_interp1d(x, y)
        f_sp = _scipy_ref(x, y)
        np.testing.assert_allclose(f_np(query), f_sp(query))


class TestLinearSpline2DIntegration:
    """Integration test – LinearSpline2D still works after the swap."""

    @pytest.fixture
    def spline(self):
        points = [(0, 0), (10, 0), (20, 10), (30, 10), (40, 0)]
        return LinearSpline2D(points)

    def test_call_at_origin(self, spline):
        x, y = spline(0.0)
        np.testing.assert_allclose((x, y), (0.0, 0.0), atol=1e-10)

    def test_call_at_length(self, spline):
        x, y = spline(spline.length)
        np.testing.assert_allclose((x, y), (40.0, 0.0), atol=1e-10)

    def test_roundtrip_cartesian_frenet(self, spline):
        original = (15.0, 3.0)
        lon, lat = spline.cartesian_to_frenet(original)
        recovered = spline.frenet_to_cartesian(lon, lat)
        np.testing.assert_allclose(recovered, original, atol=1.0)

    def test_call_midpoint(self, spline):
        mid = spline.length / 2
        x, _ = spline(mid)
        assert 0.0 <= x <= 40.0
