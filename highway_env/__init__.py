from gym.envs.registration import register

register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-merge-v0',
    entry_point='highway_env.envs:MergeEnv',
)

register(
    id='highway-roundabout-v0',
    entry_point='highway_env.envs:RoundaboutEnv',
)

register(
    id='highway-two-way-v0',
    entry_point='highway_env.envs:TwoWayEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 15}
)

register(
    id='highway-parking-v0',
    entry_point='highway_env.envs:ParkingEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20}
)
