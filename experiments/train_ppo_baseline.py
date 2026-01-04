# Reproduced from https://github.com/Farama-Foundation/HighwayEnv.git

import gymnasium as gym
import highway_env  # registers envs

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_ID = "highway-v0"

# Parallel environments (Must be under __main__)
if __name__ == "__main__":
    env = make_vec_env(
        ENV_ID,
        n_envs=8,                 # 4 or 8
        vec_env_cls=SubprocVecEnv,
        monitor_dir="runs/monitor_ppo_baseline",  # save Monitor logs per env (optional)
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="runs/tb_ppo_baseline",
        device="cpu",
    )

    model.learn(total_timesteps=50_000)
    model.save("runs/ppo_baseline_highwayenv")
