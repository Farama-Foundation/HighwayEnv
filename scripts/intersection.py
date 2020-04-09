import gym
import highway_env
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy as DQNMlp
from stable_baselines.deepq.policies import CnnPolicy as DQNCnn
from stable_baselines import PPO2, DQN


if __name__ == '__main__':
    # env = SubprocVecEnv([lambda: gym.make("intersection-v1") for i in range(1)])
    # policy_kwargs = {}
    # model = DQN(DQNCnn, env,
    #             verbose=1,
    #             policy_kwargs=policy_kwargs,
    #             batch_size=32,
    #             exploration_fraction=0.3,
    #             learning_rate=1e-4,
    #             tensorboard_log="./logs/"
    #             )
    # model.learn(total_timesteps=int(1e6))
    # model.save("deepq_intersection_v1")

    env = gym.make("intersection-v2")
    done = False
    while not done:
        # action = np.array([0, 0]) #acc, steer
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
