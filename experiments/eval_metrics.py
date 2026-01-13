import numpy as np
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO

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

N_EVAL_EPISODES = 50
SEEDS = [0, 1, 2]

EXP_ID = "cf_v2"
BASELINE_PREFIX = f"runs/models/ppo_baseline_{EXP_ID}_seed"
ATTN_PREFIX = f"runs/models/ppo_attn_{EXP_ID}_seed"

def eval_one(model_path: str, seed: int):
    env = gym.make(ENV_ID, config=ENV_CONFIG)
    model = PPO.load(model_path)

    returns = []
    lengths = []
    crashes = 0

    for ep in range(N_EVAL_EPISODES):
        obs, info = env.reset(seed=seed * 1000 + ep)
        done = truncated = False
        ep_return = 0.0
        ep_len = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

        returns.append(ep_return)
        lengths.append(ep_len)

        # highway-env sets a 'crashed' flag on the ego vehicle
        crashed = bool(env.unwrapped.vehicle.crashed)
        crashes += int(crashed)

    env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_len": float(np.mean(lengths)),
        "crash_rate": float(crashes / N_EVAL_EPISODES),
    }

def summarize(label: str, results: list[dict]):
    mean_returns = [r["mean_return"] for r in results]
    crash_rates = [r["crash_rate"] for r in results]
    mean_lens = [r["mean_len"] for r in results]

    print(f"\n{label}")
    print(f"  return:     {np.mean(mean_returns):.3f} ± {np.std(mean_returns):.3f}")
    print(f"  crash_rate: {np.mean(crash_rates):.3f} ± {np.std(crash_rates):.3f}")
    print(f"  ep_len:     {np.mean(mean_lens):.3f} ± {np.std(mean_lens):.3f}")

if __name__ == "__main__":
    baseline_results = []
    attn_results = []

    for seed in SEEDS:
        baseline_path = f"{BASELINE_PREFIX}{seed}.zip"
        attn_path = f"{ATTN_PREFIX}{seed}.zip"

        baseline_results.append(eval_one(baseline_path, seed))
        attn_results.append(eval_one(attn_path, seed))

    summarize("BASELINE", baseline_results)
    summarize("ATTENTION", attn_results)