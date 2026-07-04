from __future__ import annotations

from highway_env.envs.intersection_env import (
    ConnectedLaneMultiAgentIntersectionEnv,
    MultiAgentIntersectionEnv,
)
from highway_env.envs.common.pettingzoo_env import HighwayParallelEnv


class IntersectionParallelEnv(HighwayParallelEnv):
    """
    PettingZoo ``ParallelEnv`` for the 4-way intersection multi-agent scenario.

    Corresponds to ``intersection-multi-agent-v3`` in the Gymnasium registry
    (requires ``highway-env[multi-agent]``).

    Example::

        env = IntersectionParallelEnv(config={"controlled_vehicles": 2})
        obs, infos = env.reset(seed=42)
        while env.agents:
            actions = {a: env.action_space(a).sample() for a in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
        env.close()
    """

    metadata: dict = {
        "render_modes": ["human", "rgb_array"],
        "name": "intersection_parallel_v3",
    }

    def __init__(self, config: dict | None = None, render_mode: str | None = None) -> None:
        base = MultiAgentIntersectionEnv(config=config, render_mode=render_mode)
        super().__init__(base)


class ConnectedLaneIntersectionParallelEnv(HighwayParallelEnv):
    """
    PettingZoo ``ParallelEnv`` for the connected-lane intersection multi-agent
    scenario.

    Corresponds to ``intersection-multi-agent-v4`` in the Gymnasium registry
    (requires ``highway-env[multi-agent]``).
    """

    metadata: dict = {
        "render_modes": ["human", "rgb_array"],
        "name": "intersection_parallel_v4",
    }

    def __init__(self, config: dict | None = None, render_mode: str | None = None) -> None:
        base = ConnectedLaneMultiAgentIntersectionEnv(
            config=config, render_mode=render_mode
        )
        super().__init__(base)
