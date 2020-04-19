import gym

import highway_env


def test_highway_step():
    env = gym.make('highway-v0')

    env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1


def test_merge_step():
    env = gym.make('merge-v0')

    env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1


def test_roundabout_step():
    env = gym.make('roundabout-v0')

    env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1


def test_parking_step():
    env = gym.make('parking-v0')

    env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert action.size == 2
