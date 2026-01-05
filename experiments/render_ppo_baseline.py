import time
import gymnasium as gym
import highway_env  # registers envs
from stable_baselines3 import PPO

ENV_ID = "highway-v0"

def main():
    env = gym.make(ENV_ID, render_mode="human")
    model = PPO.load("runs/models/ppo_baseline_cf_v1")

    obs, info = env.reset()
    terminated = truncated = False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Slow it down a bit -> to watch
        time.sleep(1 / 30)

        if terminated or truncated:
            obs, info = env.reset()
            terminated = truncated = False

if __name__ == "__main__":
    main()
