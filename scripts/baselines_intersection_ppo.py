import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

import highway_env

if __name__ == '__main__':
    # Multiprocess environment
    n_cpu = 4
    env = SubprocVecEnv([lambda: gym.make('intersection-v0') for i in range(n_cpu)])

    policy_kwargs = {"net_arch": [128, 128]}
    model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./intersection_tensorboard/")
    model.learn(total_timesteps=50000)
    model.save("ppo2_intersection")

    # del model # remove to demonstrate saving and loading
    #
    # model = PPO2.load("ppo2_intersection")

    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()