import os
import sys

import gymnasium as gym


__version__ = "1.9.0"

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
    gym.register(
        id="exit-v0",
        entry_point="highway_env.envs:ExitEnv",
    )

    # highway_env.py
    gym.register(
        id="highway-v0",
        entry_point="highway_env.envs:HighwayEnv",
    )

    gym.register(
        id="highway-fast-v0",
        entry_point="highway_env.envs:HighwayEnvFast",
    )

    # intersection_env.py
    gym.register(
        id="intersection-v0",
        entry_point="highway_env.envs:IntersectionEnv",
    )

    gym.register(
        id="intersection-v1",
        entry_point="highway_env.envs:ContinuousIntersectionEnv",
    )

    gym.register(
        id="intersection-multi-agent-v0",
        entry_point="highway_env.envs:MultiAgentIntersectionEnv",
    )

    gym.register(
        id="intersection-multi-agent-v1",
        entry_point="highway_env.envs:MultiAgentIntersectionEnv",
        additional_wrappers=(MultiAgentWrapper.wrapper_spec(),),
    )

    # lane_keeping_env.py
    gym.register(
        id="lane-keeping-v0",
        entry_point="highway_env.envs:LaneKeepingEnv",
        max_episode_steps=200,
    )

    # merge_env.py
    gym.register(
        id="merge-v0",
        entry_point="highway_env.envs:MergeEnv",
    )

    # parking_env.py
    gym.register(
        id="parking-v0",
        entry_point="highway_env.envs:ParkingEnv",
    )

    gym.register(
        id="parking-ActionRepeat-v0",
        entry_point="highway_env.envs:ParkingEnvActionRepeat",
    )

    gym.register(
        id="parking-parked-v0", entry_point="highway_env.envs:ParkingEnvParkedVehicles"
    )

    # racetrack_env.py
    gym.register(
        id="racetrack-v0",
        entry_point="highway_env.envs:RacetrackEnv",
    )

    # roundabout_env.py
    gym.register(
        id="roundabout-v0",
        entry_point="highway_env.envs:RoundaboutEnv",
    )

    # two_way_env.py
    gym.register(
        id="two-way-v0", entry_point="highway_env.envs:TwoWayEnv", max_episode_steps=15
    )

    # u_turn_env.py
    gym.register(id="u-turn-v0", entry_point="highway_env.envs:UTurnEnv")


_register_highway_envs()
