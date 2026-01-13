# Reproduced from https://github.com/Farama-Foundation/HighwayEnv.git

import gymnasium as gym
import highway_env  # registers envs

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_ID = "highway-v0"

ENV_CONFIG = {
    "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": False},
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
    },
}

EXP_ID = "cf_v2"
BASE_RUN_NAME = f"ppo_baseline_{EXP_ID}"
SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 500_000

# Parallel environments (Must be under __main__)
if __name__ == "__main__":
    for seed in SEEDS:
        run_name = f"{BASE_RUN_NAME}_seed{seed}"

        # Sanity Check
        test_env = gym.make(ENV_ID, config=ENV_CONFIG)
        obs, _ = test_env.reset(seed=seed)
        print(run_name, "obs shape:", obs.shape, "action space:", test_env.action_space)
        test_env.close()

        env = make_vec_env(
            ENV_ID,
            n_envs=4,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={"config": ENV_CONFIG},
            monitor_dir=f"runs/monitor/{run_name}",
            seed=seed,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log="runs/tb",
            device="cpu",
            seed=seed,
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=run_name)
        model.save(f"runs/models/{run_name}")
        env.close()
