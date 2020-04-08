import gym
import highway_env
import numpy as np

if __name__ == '__main__':
    env = gym.make("intersection-v1")
    # env.reset()
    done = False
    while not done:
        action = np.array([0, 0]) #acc, steer
        obs, reward, done, _ = env.step(action)
        env.render()
