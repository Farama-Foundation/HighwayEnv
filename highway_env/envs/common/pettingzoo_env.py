from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import gymnasium as gym
from pettingzoo.utils.env import ParallelEnv


if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class HighwayParallelEnv(ParallelEnv):
    """
    PettingZoo ``ParallelEnv`` wrapper around any :class:`AbstractEnv` that is
    configured for multi-agent control (i.e. ``controlled_vehicles > 1`` with
    ``MultiAgentObservation`` / ``MultiAgentAction``).

    Implements **per-agent termination**: when a vehicle crashes or arrives at
    its destination, that agent is removed from :attr:`agents` while the
    remaining agents continue to act.

    Usage::

        from highway_env.envs.intersection_env import MultiAgentIntersectionEnv
        from highway_env.envs.common.pettingzoo_env import HighwayParallelEnv

        base = MultiAgentIntersectionEnv(config={"controlled_vehicles": 2})
        env = HighwayParallelEnv(base)
        obs, infos = env.reset(seed=42)

    Or via the concrete subclass::

        from highway_env.envs.intersection_pz_env import IntersectionParallelEnv
        env = IntersectionParallelEnv(config={"controlled_vehicles": 2})

    Parameters
    ----------
    env:
        A fully-initialised :class:`AbstractEnv` with ``MultiAgentObservation``
        and ``MultiAgentAction`` configured.
    """

    metadata: dict = {
        "render_modes": ["human", "rgb_array"],
        "name": "highway_parallel_v0",
    }

    def __init__(self, env: AbstractEnv) -> None:
        super().__init__()
        self._env = env
        n = len(env.controlled_vehicles)
        self.possible_agents: list[str] = [f"agent_{i}" for i in range(n)]
        self.agents: list[str] = self.possible_agents[:]
        # Stable index lookup: agent name → index in possible_agents / controlled_vehicles
        self._agent_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(self.possible_agents)
        }

    # ------------------------------------------------------------------
    # PettingZoo required methods
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gym.Space:
        """Return the observation space for *agent*.

        The base env's ``observation_space`` is a ``spaces.Tuple``; we index
        into it to return the per-agent sub-space.
        """
        idx = self._agent_to_idx[agent]
        return self._env.observation_space.spaces[idx]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.Space:
        """Return the action space for *agent*."""
        idx = self._agent_to_idx[agent]
        return self._env.action_space.spaces[idx]

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset the environment.

        Returns
        -------
        observations:
            ``{agent_name: obs_array}`` for all agents in :attr:`possible_agents`.
        infos:
            ``{agent_name: info_dict}`` (shared info dict for now).
        """
        obs_tuple, info = self._env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        observations = {
            agent: obs_tuple[self._agent_to_idx[agent]] for agent in self.agents
        }
        infos = {agent: info for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, int | float]
    ) -> tuple[dict, dict, dict, dict, dict]:
        """Step the environment with one action per active agent.

        ``actions`` must contain an entry for every agent currently in
        :attr:`agents`. Agents that have already terminated must NOT be
        included (PettingZoo contract).

        Returns
        -------
        observations, rewards, terminations, truncations, infos:
            All keyed by agent name, covering only the agents that were
            *active at the start of this step* (i.e. the pre-step
            :attr:`agents` list).
        """
        # Build the full action tuple the base env expects.
        # Agents no longer in self.agents have already been removed in a
        # previous step, so we supply a zero/idle action for them to keep
        # the tuple length equal to len(possible_agents).
        action_tuple = tuple(
            actions[agent] if agent in actions else self._idle_action(agent)
            for agent in self.possible_agents
        )

        obs_tuple, _reward, _terminated, truncated, info = self._env.step(action_tuple)

        # Prefer per-agent breakdown from info (set by IntersectionEnv._info)
        agents_rewards: tuple = info.get(
            "agents_rewards", (_reward,) * len(self.possible_agents)
        )
        agents_terminated: tuple = info.get(
            "agents_terminated", (_terminated,) * len(self.possible_agents)
        )

        # Build per-agent dicts for agents that were active this step
        active_agents = list(self.agents)  # snapshot before mutation
        observations: dict = {}
        rewards: dict = {}
        terminations: dict = {}
        truncations: dict = {}
        infos: dict = {}

        for agent in active_agents:
            i = self._agent_to_idx[agent]
            observations[agent] = obs_tuple[i]
            rewards[agent] = float(agents_rewards[i])
            terminations[agent] = bool(agents_terminated[i])
            truncations[agent] = bool(truncated)
            infos[agent] = info

        # Remove agents that are done from self.agents (PettingZoo contract)
        self.agents = [
            agent
            for agent in self.agents
            if not terminations.get(agent, False) and not truncations.get(agent, False)
        ]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Delegate rendering to the inner environment."""
        return self._env.render()

    def close(self) -> None:
        """Close the inner environment."""
        return self._env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _idle_action(self, agent: str):
        """Return a safe no-op action for a terminated agent.

        For ``DiscreteMetaAction`` this is index 1 (IDLE).
        For continuous actions this is a zero vector.
        The underlying vehicle is already inert (crashed), so the exact
        value does not affect physics.
        """
        space = self.action_space(agent)
        # Discrete: return IDLE (index 1 in DiscreteMetaAction)
        if hasattr(space, "n"):
            return min(1, space.n - 1)
        # Continuous: return zeros
        import numpy as np

        return np.zeros(space.shape, dtype=space.dtype)
