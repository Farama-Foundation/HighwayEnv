from gym.envs.registration import register

register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='merge-v0',
    entry_point='highway_env.envs:MergeEnv',
)

register(
    id='roundabout-v0',
    entry_point='highway_env.envs:RoundaboutEnv',
)

register(
    id='two-way-v0',
    entry_point='highway_env.envs:TwoWayEnv',
    max_episode_steps=15
)

register(
    id='parking-v0',
    entry_point='highway_env.envs:ParkingEnv',
    max_episode_steps=20
)
