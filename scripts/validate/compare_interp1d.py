"""Benchmark: numpy_interp1d vs scipy.interpolate.interp1d(fill_value="extrapolate")."""

import timeit

import numpy as np
from scipy import interpolate

from highway_env.road.spline import numpy_interp1d


DATA_SIZES = [2, 10, 100, 1_000, 10_000]
QUERY_SIZES = [1, 10, 100, 1_000, 10_000]
N_REPEAT = 100_000
BENCHMARK_SEED = 42


def _make_data(n: int, rng: np.random.Generator):
    x = np.sort(rng.uniform(0, 1000, size=n))
    y = rng.standard_normal(n)
    return x, y


def _make_query(n: int, x: np.ndarray, rng: np.random.Generator):
    lo, hi = x[0] - 50, x[-1] + 50
    return rng.uniform(lo, hi, size=n)


def bench_construction():
    """Constructor check: is numpy_interp1d fast enough to construct."""
    rng = np.random.default_rng(BENCHMARK_SEED)
    print("=" * 72)
    print("CONSTRUCTION TIME (build the interpolator)")
    print(f"{'data pts':>10}  {'scipy (ms)':>12}  {'numpy (ms)':>12}  {'speedup':>8}")
    print("-" * 72)
    for n_data in DATA_SIZES:
        x, y = _make_data(n_data, rng)
        ns = {
            "interpolate": interpolate,
            "numpy_interp1d": numpy_interp1d,
            "x": x,
            "y": y,
        }
        t_sp = timeit.Timer(
            "interpolate.interp1d(x, y, fill_value='extrapolate')", globals=ns
        ).timeit(N_REPEAT)
        t_np = timeit.Timer("numpy_interp1d(x, y)", globals=ns).timeit(N_REPEAT)
        ms_sp = t_sp / N_REPEAT * 1000
        ms_np = t_np / N_REPEAT * 1000
        print(f"{n_data:>10}  {ms_sp:>12.4f}  {ms_np:>12.4f}  {ms_sp / ms_np:>7.1f}x")


def bench_evaluation():
    """Evaluation check: is numpy_interp1d fast enough in evaluation."""
    rng = np.random.default_rng(BENCHMARK_SEED)
    print()
    print("=" * 72)
    print("EVALUATION TIME (call the interpolator)")
    print(
        f"{'data pts':>10}  {'query pts':>10}  "
        f"{'scipy (ms)':>12}  {'numpy (ms)':>12}  {'speedup':>8}"
    )
    print("-" * 72)
    for n_data in DATA_SIZES:
        x, y = _make_data(n_data, rng)
        f_sp = interpolate.interp1d(x, y, fill_value="extrapolate")
        f_np = numpy_interp1d(x, y)
        for n_query in QUERY_SIZES:
            q = _make_query(n_query, x, rng)
            ns = {"f_sp": f_sp, "f_np": f_np, "q": q}
            t_sp = timeit.Timer("f_sp(q)", globals=ns).timeit(N_REPEAT)
            t_np = timeit.Timer("f_np(q)", globals=ns).timeit(N_REPEAT)
            ms_sp = t_sp / N_REPEAT * 1000
            ms_np = t_np / N_REPEAT * 1000
            print(
                f"{n_data:>10}  {n_query:>10}  "
                f"{ms_sp:>12.4f}  {ms_np:>12.4f}  {ms_sp / ms_np:>7.1f}x"
            )


def validate_correctness():
    """Sanify check: are they equivalent."""
    rng = np.random.default_rng(BENCHMARK_SEED)
    print()
    print("=" * 72)
    print("CORRECTNESS VALIDATION (assert_allclose scipy vs numpy)")
    print("-" * 72)
    for n_data in DATA_SIZES:
        x, y = _make_data(n_data, rng)
        f_sp = interpolate.interp1d(x, y, fill_value="extrapolate")
        f_np = numpy_interp1d(x, y)
        for n_query in QUERY_SIZES:
            q = _make_query(n_query, x, rng)
            res_sp = f_sp(q)
            res_np = f_np(q)
            np.testing.assert_allclose(res_np, res_sp, atol=1e-10)
            max_diff = np.max(np.abs(res_np - res_sp))
            print(
                f"  data={n_data:>5}  query={n_query:>5}  "
                f"max_abs_diff={max_diff:.2e}  PASS"
            )
    print("\nAll correctness checks passed.\n")


if __name__ == "__main__":
    validate_correctness()
    bench_construction()
    bench_evaluation()
