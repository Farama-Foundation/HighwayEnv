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
