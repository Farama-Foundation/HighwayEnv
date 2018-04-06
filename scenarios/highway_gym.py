import gym
import highway_env

env = gym.make('highway-v0')
obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    print('reward = {}, done = {}'.format(reward, done))
env.close()
