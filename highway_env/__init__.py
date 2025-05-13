import os
import sys

from gymnasium.envs.registration import register


__version__ = "1.10.1"

try:
    from farama_notifications import notifications

    if "highway_env" in notifications and __version__ in notifications["gymnasium"]:
        print(notifications["highway_env"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass

# Hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _register_highway_envs():
    """Import the envs module so that envs register themselves."""

    from highway_env.envs.common.abstract import MultiAgentWrapper

    # exit_env.py
    register(
        id="exit-v0",
        entry_point="highway_env.envs.exit_env:ExitEnv",
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
        id="intersection-multi-agent-v0",
        entry_point="highway_env.envs.intersection_env:MultiAgentIntersectionEnv",
    )

    register(
        id="intersection-multi-agent-v1",
        entry_point="highway_env.envs.intersection_env:MultiAgentIntersectionEnv",
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
        id="racetrack-large-v0",
        entry_point="highway_env.envs.racetrack_env:RacetrackEnvLarge",
    )
    register(
        id="racetrack-oval-v0",
        entry_point="highway_env.envs.racetrack_env:RacetrackEnvOval",
    )

    # roundabout_env.py
    register(
        id="roundabout-v0",
        entry_point="highway_env.envs.roundabout_env:RoundaboutEnv",
    )

    # two_way_env.py
    register(
        id="two-way-v0",
        entry_point="highway_env.envs.two_way_env:TwoWayEnv",
        max_episode_steps=15,
    )

    # u_turn_env.py
    register(id="u-turn-v0", entry_point="highway_env.envs.u_turn_env:UTurnEnv")


_register_highway_envs()
