import time
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

MODEL_PATH = "runs/models/ppo_attn_cf_v1.zip"

def main():
    env = gym.make(ENV_ID, config=ENV_CONFIG, render_mode="human")
    model = PPO.load(MODEL_PATH)

    obs, info = env.reset()
    attn_log = []
    dx_log = []

    for t in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract attention weights (batch=1)
        attn = model.policy.features_extractor.last_attention
        if attn is not None:
            attn_log.append(attn.squeeze(0).cpu().numpy())  # (N_nei,)

        # Log neighbour relative x positions for context (slots 1..4)
        # features: [presence, x, y, vx, vy] => x is column 1
        dx_log.append(obs[1:, 1].copy())

        time.sleep(1/30)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    attn_log = np.array(attn_log)
    dx_log = np.array(dx_log)

    np.save("runs/attention_weights.npy", attn_log)
    np.save("runs/neighbour_dx.npy", dx_log)
    print("Saved:", attn_log.shape, dx_log.shape)

if __name__ == "__main__":
    main()
