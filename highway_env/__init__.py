# Hide pygame support prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


from gymnasium.envs.registration import register


def register_highway_envs():
    """Import the envs module so that envs register themselves."""

    # exit_env.py
    register(
        id='exit-v0',
        entry_point='highway_env.envs:ExitEnv',
    )

    # highway_env.py
    register(
        id='highway-v0',
        entry_point='highway_env.envs:HighwayEnv',
    )

    register(
        id='highway-fast-v0',
        entry_point='highway_env.envs:HighwayEnvFast',
    )

    # intersection_env.py
    register(
        id='intersection-v0',
        entry_point='highway_env.envs:IntersectionEnv',
    )

    register(
        id='intersection-v1',
        entry_point='highway_env.envs:ContinuousIntersectionEnv',
    )

    register(
        id='intersection-multi-agent-v0',
        entry_point='highway_env.envs:MultiAgentIntersectionEnv',
    )

    register(
        id='intersection-multi-agent-v1',
        entry_point='highway_env.envs:TupleMultiAgentIntersectionEnv',
    )

    # lane_keeping_env.py
    register(
        id='lane-keeping-v0',
        entry_point='highway_env.envs:LaneKeepingEnv',
        max_episode_steps=200
    )

    # merge_env.py
    register(
        id='merge-v0',
        entry_point='highway_env.envs:MergeEnv',
    )

    # parking_env.py
    register(
        id='parking-v0',
        entry_point='highway_env.envs:ParkingEnv',
    )

    register(
        id='parking-ActionRepeat-v0',
        entry_point='highway_env.envs:ParkingEnvActionRepeat'
    )

    register(
        id='parking-parked-v0',
        entry_point='highway_env.envs:ParkingEnvParkedVehicles'
    )

    # racetrack_env.py
    register(
        id='racetrack-v0',
        entry_point='highway_env.envs:RacetrackEnv',
    )

    # roundabout_env.py
    register(
        id='roundabout-v0',
        entry_point='highway_env.envs:RoundaboutEnv',
    )

    # two_way_env.py
    register(
        id='two-way-v0',
        entry_point='highway_env.envs:TwoWayEnv',
        max_episode_steps=15
    )

    # u_turn_env.py
    register(
        id='u-turn-v0',
        entry_point='highway_env.envs:UTurnEnv'
    )

