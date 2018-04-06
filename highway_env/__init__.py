from gym.envs.registration import register

register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)