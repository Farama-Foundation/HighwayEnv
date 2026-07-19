"""Tests for PettingZoo ParallelEnv integration.

Skipped automatically if pettingzoo is not installed
(e.g. in base CI without the [multi-agent] extra).
"""

from __future__ import annotations

import pytest


# Skip entire module if pettingzoo is not available
pytest.importorskip(
    "pettingzoo.test",
    reason="pettingzoo not installed; run with: pip install highway-env[multi-agent]",
)
from pettingzoo.test import parallel_api_test  # noqa: E402

import highway_env  # noqa: E402,F401 — triggers env registration
from highway_env.envs.intersection_pz_env import (  # noqa: E402
    ConnectedLaneIntersectionParallelEnv,
    IntersectionParallelEnv,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def intersection_env():
    env = IntersectionParallelEnv(config={"controlled_vehicles": 2, "duration": 10})
    yield env
    env.close()


@pytest.fixture
def connected_lane_env():
    env = ConnectedLaneIntersectionParallelEnv(
        config={"controlled_vehicles": 2, "duration": 10}
    )
    yield env
    env.close()


# -----------------------------------------------------------------------
# PettingZoo API conformance
# -----------------------------------------------------------------------


def test_parallel_api_test_intersection():
    """PettingZoo's built-in API conformance checker must pass."""
    env = IntersectionParallelEnv(config={"controlled_vehicles": 2, "duration": 5})
    parallel_api_test(env, num_cycles=10)
    env.close()


def test_parallel_api_test_connected_lane():
    """Same for the connected-lane variant."""
    env = ConnectedLaneIntersectionParallelEnv(
        config={"controlled_vehicles": 2, "duration": 5}
    )
    parallel_api_test(env, num_cycles=10)
    env.close()


# -----------------------------------------------------------------------
# Observation / action space correctness
# -----------------------------------------------------------------------


def test_reset_returns_agent_dict(intersection_env):
    """reset() must return dicts keyed by agent name."""
    obs, infos = intersection_env.reset(seed=42)
    assert isinstance(obs, dict)
    assert isinstance(infos, dict)
    assert set(obs.keys()) == set(intersection_env.possible_agents)
    assert set(infos.keys()) == set(intersection_env.possible_agents)


def test_obs_in_observation_space(intersection_env):
    """Every observation must lie within the declared observation space."""
    obs, _ = intersection_env.reset(seed=0)
    while intersection_env.agents:
        actions = {
            a: intersection_env.action_space(a).sample()
            for a in intersection_env.agents
        }
        obs, rewards, terminations, truncations, infos = intersection_env.step(actions)
        for agent, agent_obs in obs.items():
            assert intersection_env.observation_space(agent).contains(
                agent_obs
            ), f"{agent}: obs not in observation_space"


def test_possible_agents_is_stable(intersection_env):
    """possible_agents must not change across the episode."""
    intersection_env.reset(seed=1)
    initial_possible = list(intersection_env.possible_agents)
    while intersection_env.agents:
        actions = {
            a: intersection_env.action_space(a).sample()
            for a in intersection_env.agents
        }
        intersection_env.step(actions)
    assert intersection_env.possible_agents == initial_possible


# -----------------------------------------------------------------------
# Per-agent termination semantics
# -----------------------------------------------------------------------


def test_terminated_agents_removed_from_agents(intersection_env):
    """Agents flagged as terminated must NOT appear in self.agents next step."""
    intersection_env.reset(seed=2)
    while intersection_env.agents:
        pre_step_agents = set(intersection_env.agents)
        actions = {
            a: intersection_env.action_space(a).sample()
            for a in intersection_env.agents
        }
        _, _, terminations, truncations, _ = intersection_env.step(actions)

        for agent in pre_step_agents:
            if terminations.get(agent, False) or truncations.get(agent, False):
                assert (
                    agent not in intersection_env.agents
                ), f"{agent} terminated/truncated but still in env.agents"


def test_episode_ends_when_all_agents_done(intersection_env):
    """Episode loop must naturally exit when all agents terminate."""
    intersection_env.reset(seed=3)
    steps = 0
    max_steps = 500
    while intersection_env.agents and steps < max_steps:
        actions = {
            a: intersection_env.action_space(a).sample()
            for a in intersection_env.agents
        }
        intersection_env.step(actions)
        steps += 1
    # If we reach here without error, the loop terminated correctly
    assert steps < max_steps, "Episode never ended — env may be stuck"


# -----------------------------------------------------------------------
# Seeding / reproducibility
# -----------------------------------------------------------------------


def test_reset_with_same_seed_is_reproducible():
    """Two resets with the same seed must produce identical initial observations."""
    import numpy as np

    env1 = IntersectionParallelEnv(config={"controlled_vehicles": 2, "duration": 5})
    env2 = IntersectionParallelEnv(config={"controlled_vehicles": 2, "duration": 5})

    obs1, _ = env1.reset(seed=99)
    obs2, _ = env2.reset(seed=99)

    for agent in obs1:
        np.testing.assert_array_equal(obs1[agent], obs2[agent])

    env1.close()
    env2.close()


# -----------------------------------------------------------------------
# Deprecation warning for MultiAgentWrapper
# -----------------------------------------------------------------------


def test_multi_agent_wrapper_deprecation_warning():
    """Using MultiAgentWrapper must raise DeprecationWarning."""
    import warnings

    import gymnasium as gym

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = gym.make("intersection-multi-agent-v1")
        env.close()

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert (
        len(deprecation_warnings) > 0
    ), "Expected DeprecationWarning from MultiAgentWrapper but none was raised"
    assert any(
        "MultiAgentWrapper" in str(w.message) for w in deprecation_warnings
    ), "Expected a DeprecationWarning mentioning MultiAgentWrapper"
