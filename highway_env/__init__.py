import os
import sys

from gymnasium.envs.registration import register, registry


__version__ = "1.12.0.dev1"

try:
    from farama_notifications import notifications

    if "highway_env" in notifications and __version__ in notifications["highway_env"]:
        print(notifications["highway_env"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass

# Hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _register_highway_envs():
    """Import the envs module so that envs register themselves.

    This function is idempotent: calling it multiple times (e.g. when
    gymnasium resolves a ``"highway_env:env-id"`` spec in a subprocess)
    will not raise duplicate-registration errors.
    """
    # Skip if environments are already registered (idempotent)
    if "highway-v0" in registry:
        return

    from highway_env.envs.common.abstract import MultiAgentWrapper

    # exit_env.py
    register(
        id="exit-v0",
        entry_point="highway_env.envs.exit_env:ExitEnv",
    )
    register(
        id="exit-v1",
        entry_point="highway_env.envs.exit_env:ConnectedLaneExitEnv",
    )

    # highway_env.py
    register(
        id="highway-v0",
        entry_point="highway_env.envs.highway_env:HighwayEnv",
    )

    register(
        id="highway-fast-v0",
        entry_point="highway_env.envs.highway_env:HighwayEnvFast",
    )

    # intersection_env.py
    register(
        id="intersection-v0",
        entry_point="highway_env.envs.intersection_env:IntersectionEnv",
    )

    register(
        id="intersection-v1",
        entry_point="highway_env.envs.intersection_env:ContinuousIntersectionEnv",
    )
    register(
        id="intersection-v2",
        entry_point="highway_env.envs.intersection_env:ConnectedLaneIntersectionEnv",
    )

    register(
        id="intersection-multi-agent-v0",
        entry_point="highway_env.envs.intersection_env:MultiAgentIntersectionEnv",
    )

    register(
        id="intersection-multi-agent-v1",
        entry_point="highway_env.envs.intersection_env:MultiAgentIntersectionEnv",
        additional_wrappers=(MultiAgentWrapper.wrapper_spec(),),
    )
    register(
        id="intersection-multi-agent-v2",
        entry_point="highway_env.envs.intersection_env:ConnectedLaneMultiAgentIntersectionEnv",
        additional_wrappers=(MultiAgentWrapper.wrapper_spec(),),
    )

    # lane_keeping_env.py
    register(
        id="lane-keeping-v0",
        entry_point="highway_env.envs.lane_keeping_env:LaneKeepingEnv",
        max_episode_steps=200,
    )

    # merge_env.py
    register(
        id="merge-v0",
        entry_point="highway_env.envs.merge_env:MergeEnv",
    )
    register(
        id="merge-v1",
        entry_point="highway_env.envs.merge_env:ConnectedLaneMergeEnv",
    )
    register(
        id="merge-generic-v0",
        entry_point="highway_env.envs.merge_env:MergeGenericEnv",
    )
    register(
        id="merge-generic-v1",
        entry_point="highway_env.envs.merge_env:ConnectedLaneMergeGenericEnv",
    )

    # parking_env.py
    register(
        id="parking-v0",
        entry_point="highway_env.envs.parking_env:ParkingEnv",
    )

    register(
        id="parking-ActionRepeat-v0",
        entry_point="highway_env.envs.parking_env:ParkingEnvActionRepeat",
    )

    register(
        id="parking-parked-v0",
        entry_point="highway_env.envs.parking_env:ParkingEnvParkedVehicles",
    )

    # racetrack_env.py
    register(
        id="racetrack-v0",
        entry_point="highway_env.envs.racetrack_env:RacetrackEnv",
    )
    register(
        id="racetrack-v1",
        entry_point="highway_env.envs.racetrack_env:ConnectedLaneRacetrackEnv",
    )
    register(
        id="racetrack-large-v0",
        entry_point="highway_env.envs.racetrack_env:RacetrackEnvLarge",
    )
    register(
        id="racetrack-large-v1",
        entry_point="highway_env.envs.racetrack_env:ConnectedLaneRacetrackEnvLarge",
    )
    register(
        id="racetrack-oval-v0",
        entry_point="highway_env.envs.racetrack_env:RacetrackEnvOval",
    )
    register(
        id="racetrack-oval-v1",
        entry_point="highway_env.envs.racetrack_env:ConnectedLaneRacetrackEnvOval",
    )

    # roundabout_env.py
    register(
        id="roundabout-v0",
        entry_point="highway_env.envs.roundabout_env:RoundaboutEnv",
    )
    register(
        id="roundabout-v1",
        entry_point="highway_env.envs.roundabout_env:ConnectedLaneRoundaboutEnv",
    )
    register(
        id="roundabout-generic-v0",
        entry_point="highway_env.envs.roundabout_env:RoundaboutGenericEnv",
    )
    register(
        id="roundabout-generic-v1",
        entry_point="highway_env.envs.roundabout_env:ConnectedLaneRoundaboutGenericEnv",
    )

    # two_way_env.py
    register(
        id="two-way-v0",
        entry_point="highway_env.envs.two_way_env:TwoWayEnv",
        max_episode_steps=15,
    )

    # u_turn_env.py
    register(id="u-turn-v0", entry_point="highway_env.envs.u_turn_env:UTurnEnv")
    register(
        id="u-turn-v1",
        entry_point="highway_env.envs.u_turn_env:ConnectedLaneUTurnEnv",
    )

    _register_pettingzoo_envs()

def _register_pettingzoo_envs() -> None:
    """Register PettingZoo ParallelEnv IDs (skipped silently if pettingzoo not installed).
    IDs continue the ``intersection-multi-agent-vN`` series from v2:
      - v3: IntersectionParallelEnv (standard)
      - v4: ConnectedLaneIntersectionParallelEnv
    Note: These are PettingZoo ParallelEnv instances, NOT gymnasium.Env subclasses.
    ``gymnasium.make("intersection-multi-agent-v3")`` will NOT work; use direct import::
        from highway_env.envs.intersection_pz_env import IntersectionParallelEnv
        env = IntersectionParallelEnv()
    """
    try:
        from pettingzoo.utils.env import ParallelEnv  # noqa: F401
    except ImportError:
        return  # pettingzoo not installed; skip silently
    if "intersection-multi-agent-v3" not in registry:
        register(
            id="intersection-multi-agent-v3",
            entry_point="highway_env.envs.intersection_pz_env:IntersectionParallelEnv",
        )
    if "intersection-multi-agent-v4" not in registry:
        register(
            id="intersection-multi-agent-v4",
            entry_point=(
                "highway_env.envs.intersection_pz_env"
                ":ConnectedLaneIntersectionParallelEnv"
            ),
        )

_register_highway_envs()
