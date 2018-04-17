from __future__ import division, print_function
import gym

import highway_env


def test_highway_step():
    env = gym.make('highway-v0')

    env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert obs
    assert reward
    assert not done


def test_merge_step():
    env = gym.make('highway-merge-v0')

    env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert obs
    assert reward
    assert not done
