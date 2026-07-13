"""Test that highway-env environments work with multiprocessing (forkserver/spawn).

When using SubprocVecEnv from stable-baselines3 or similar vectorized
environment wrappers, child processes are started via ``forkserver`` or
``spawn`` and do **not** inherit the parent's ``import highway_env``.

Gymnasium's ``module:env_name`` syntax (e.g. ``"highway_env:highway-v0"``)
triggers an import of the module in the subprocess, which registers the
environments on demand.

See: https://github.com/Farama-Foundation/HighwayEnv/issues/648
"""

import multiprocessing as mp

import gymnasium as gym
import pytest


def _make_env_in_subprocess(env_id: str, result_queue: mp.Queue) -> None:
    """Create and step an environment inside a subprocess (no prior import of highway_env)."""
    try:
        env = gym.make(env_id)
        obs, _info = env.reset()
        _obs, _reward, _terminated, _truncated, _info = env.step(
            env.action_space.sample()
        )
        env.close()
        result_queue.put(("ok", str(type(obs))))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


@pytest.mark.parametrize(
    "env_id",
    [
        "highway_env:highway-v0",
        "highway_env:highway-fast-v0",
        "highway_env:merge-v0",
        "highway_env:roundabout-v0",
        "highway_env:intersection-v0",
        "highway_env:parking-v0",
    ],
)
@pytest.mark.parametrize("start_method", ["forkserver", "spawn"])
def test_env_in_subprocess(env_id: str, start_method: str) -> None:
    """Environments should be creatable in forkserver/spawn subprocesses via module:name syntax."""
    if start_method not in mp.get_all_start_methods():
        pytest.skip(f"{start_method} not available on this platform")

    ctx = mp.get_context(start_method)
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_make_env_in_subprocess, args=(env_id, q))
    p.start()
    p.join(timeout=30)

    assert not q.empty(), "Subprocess produced no result (likely crashed)"
    status, detail = q.get()
    assert status == "ok", f"Subprocess failed: {detail}"
