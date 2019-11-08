import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy as DQNMlp
from stable_baselines import PPO2, DQN

import highway_env

cfg = {
    "environment": "intersection-v0",
    "--processes": 1,
    "--steps": 10e4,
    "--n_steps": 128,
    "--learning_rate": 0.5e-3,
    "--batch_size": 64,
    "--train": True,
    "--test": False
}

if __name__ == '__main__':
    # Multiprocess environment
    env = SubprocVecEnv([lambda: gym.make(cfg["environment"]) for i in range(int(cfg["--processes"]))])

    # if cfg["--train"]:
    #     policy_kwargs = {"net_arch": [512, 512]}
    #     model = PPO2(MlpPolicy, env,
    #                  verbose=1,
    #                  policy_kwargs=policy_kwargs,
    #                  n_steps=cfg["--n_steps"],
    #                  learning_rate=cfg["--learning_rate"],
    #                  tensorboard_log="./logs/")
    #     model.learn(total_timesteps=int(cfg["--steps"]))
    #     model.save("ppo2_intersection")
    #
    # if cfg["--test"]:
    #     model = PPO2.load("ppo2_intersection")
    #     obs = env.reset()
    #     while True:
    #         action, _states = model.predict(obs)
    #         obs, rewards, dones, info = env.step(action)
    #         env.render()

    if cfg["--train"]:
        policy_kwargs = {}
        model = DQN(DQNMlp, env,
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    batch_size=cfg["--batch_size"],
                    exploration_fraction=0.3,
                    learning_rate=cfg["--learning_rate"],
                    tensorboard_log="./logs/")
        model.learn(total_timesteps=int(cfg["--steps"]))
        model.save("deepq_intersection")
