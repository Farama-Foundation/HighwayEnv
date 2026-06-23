"""
Benchmark render-mode FPS across highway-env environments.

Measures wall-clock frames-per-second for three render modes:
  - None          (headless, no rendering at all)
  - "rgb_array"   (offscreen rendering, returns pixel buffer)
  - "human"       (on-screen window rendering)

The first two set OFFSCREEN_RENDERING=1 so pygame never tries to open a display.

Run:
    uv run python scripts/regression_test/bench_render_fps.py
    uv run python scripts/regression_test/bench_render_fps.py --envs highway-v0 intersection-v0
    uv run python scripts/regression_test/bench_render_fps.py --steps 200 --repeat 50
    uv run python scripts/regression_test/bench_render_fps.py --name my_run
"""

import argparse
import json
import os
import platform
import time
import warnings

import gymnasium as gym
import numpy as np
from tqdm.rich import trange

from highway_env import __version__ as env_ver


SEED = 42
DEFAULT_STEPS = 100
DEFAULT_REPEATS = 100
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_NAME = "bench_render_fps"
DEFAULT_ENVS = [
    "highway-v0",
    "intersection-v0",
    "roundabout-v0",
    "merge-v0",
    "racetrack-v0",
]
RENDER_MODES: list[dict] = [
    {"render_mode": None, "offscreen": True, "label": "None"},
    {"render_mode": "rgb_array", "offscreen": True, "label": "rgb_array"},
    {"render_mode": "human", "offscreen": False, "label": "human"},
]

warnings.filterwarnings("ignore")
print(f"HighwayEnv Version: {env_ver}")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_episode(
    env_id: str,
    render_mode: str | None,
    offscreen: bool,
    n_steps: int,
    seed: int,
) -> float:
    """Run one episode and return elapsed seconds."""
    env = gym.make(
        env_id,
        render_mode=render_mode,
        config={"offscreen_rendering": offscreen, "real_time_rendering": False},
    )

    env.reset(seed=seed)
    action_space = env.action_space
    action_space.seed(seed)

    t_start = time.perf_counter()
    for _ in range(n_steps):
        action = action_space.sample()
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if render_mode == "rgb_array":
            env.render()
        if terminated or truncated:
            env.reset(seed=seed)
    elapsed = time.perf_counter() - t_start

    env.close()
    time.sleep(1)
    return elapsed


def bench_render_mode(
    env_id: str,
    render_mode: str | None,
    offscreen: bool,
    n_steps: int,
    n_repeat: int,
) -> dict:
    """Repeat run_episode n_repeat times (seed increments each iter) and return stats."""
    old_val = os.environ.get("OFFSCREEN_RENDERING")
    os.environ["OFFSCREEN_RENDERING"] = "1" if offscreen else "0"
    try:
        timings = [
            run_episode(env_id, render_mode, offscreen, n_steps, seed=SEED + i)
            for i in trange(n_repeat)
        ]
    finally:
        if old_val is None:
            os.environ.pop("OFFSCREEN_RENDERING", None)
        else:
            os.environ["OFFSCREEN_RENDERING"] = old_val

    fps_values = [n_steps / t for t in timings]
    return {
        "env": env_id,
        "render_mode": str(render_mode),
        "offscreen": offscreen,
        "steps": n_steps,
        "n_repeat": n_repeat,
        "mean_fps": round(float(np.mean(fps_values)), 2),
        "std_fps": round(float(np.std(fps_values)), 2),
        "min_fps": round(float(np.min(fps_values)), 2),
        "max_fps": round(float(np.max(fps_values)), 2),
        "mean_elapsed_s": round(float(np.mean(timings)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark render-mode FPS")
    parser.add_argument(
        "--envs", nargs="+", default=DEFAULT_ENVS, help="Environment IDs to benchmark"
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS, help="Steps per episode"
    )
    parser.add_argument(
        "--repeat", type=int, default=DEFAULT_REPEATS, help="Repeats per configuration"
    )
    parser.add_argument(
        "--name", default=DEFAULT_NAME, help="Output filename (without extension)"
    )
    args = parser.parse_args()

    result_file = os.path.join(RESULTS_DIR, f"{args.name}.json")

    results: list[dict] = []

    header = (
        f"{'env':>25}  {'render_mode':>12}  "
        f"{'repeat':>7}  {'mean_fps':>9}  {'std_fps':>8}  "
        f"{'min_fps':>8}  {'max_fps':>8}"
    )
    print("=" * len(header))
    print("RENDER-MODE FPS BENCHMARK")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for env_id in args.envs:
        for mode_cfg in RENDER_MODES:
            result = bench_render_mode(
                env_id,
                mode_cfg["render_mode"],
                mode_cfg["offscreen"],
                args.steps,
                args.repeat,
            )
            results.append(result)
            print(
                f"{result['env']:>25}  {mode_cfg['label']:>12}  "
                f"{result['n_repeat']:>7}  {result['mean_fps']:>9.2f}  "
                f"{result['std_fps']:>8.2f}  {result['min_fps']:>8.2f}  "
                f"{result['max_fps']:>8.2f}"
            )
            time.sleep(10)
        print()

    output = {
        "meta": {
            "seed": SEED,
            "steps": args.steps,
            "n_repeat": args.repeat,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
