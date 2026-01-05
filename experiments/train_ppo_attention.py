import gymnasium as gym
import highway_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from attention_extractor import KinematicAttentionExtractor

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

RUN_NAME = "ppo_attn_cf_v1"
SEED = 0

if __name__ == "__main__":
    # Sanity check
    test_env = gym.make(ENV_ID, config=ENV_CONFIG)
    obs, _ = test_env.reset()
    print("obs shape:", obs.shape)
    print("action space:", test_env.action_space)
    test_env.close()

    env = make_vec_env(
        ENV_ID,
        n_envs=4,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": ENV_CONFIG},
        monitor_dir=f"runs/monitor/{RUN_NAME}",
        seed=SEED,
    )

    policy_kwargs = dict(
        features_extractor_class=KinematicAttentionExtractor,
        features_extractor_kwargs=dict(features_dim=128, d_model=32),
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="runs/tb",
        device="cpu",
        seed=SEED,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=50_000, tb_log_name=RUN_NAME)
    model.save(f"runs/models/{RUN_NAME}")
