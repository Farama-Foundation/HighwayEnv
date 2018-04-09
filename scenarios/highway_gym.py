from __future__ import division, print_function
import gym
from gym.wrappers.monitor import Monitor

import highway_env

env = gym.make('highway-v0')
monitor = Monitor(env, 'out', force=True)
obs = env.reset()

done = False
monitor.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = monitor.step(action)
    env.render()
    print('reward = {}, done = {}'.format(reward, done))
monitor.close()
env.close()
