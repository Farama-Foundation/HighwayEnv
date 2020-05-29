import gym
import highway_env
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy as DQNMlp
from stable_baselines.deepq.policies import CnnPolicy as DQNCnn
from stable_baselines import PPO2, DQN
from stable_baselines.common.policies import CnnPolicy

if __name__ == '__main__':
    env = SubprocVecEnv([lambda: gym.make("intersection-v2") for i in range(2)])
    model = PPO2(CnnPolicy,
                 env,
                 verbose=1,
                 tensorboard_log="./logs/"
                )
    model.learn(total_timesteps=int(1e6))
    model.save("ppo2_intersection_v2")

    # env = gym.make("intersection-v2")
    # done = False
    # env.reset()
    # while not done:
    #     # action = np.array([0, 0]) #acc, steer
    #     action = env.action_space.sample()
    #     obs, reward, done, _ = env.step(action)
    #     env.render()
