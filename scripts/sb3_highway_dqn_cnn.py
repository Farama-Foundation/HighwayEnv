import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

import highway_env


def train_env():
    env = gym.make('highway-fast-v0')
    env.configure({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
    })
    env.reset()
    return env


def test_env():
    env = train_env()
    env.configure({"policy_frequency": 15, "duration": 20})
    env.reset()
    return env


if __name__ == '__main__':
    # Train
    model = DQN('CnnPolicy', DummyVecEnv([train_env]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log="highway_cnn/")
    model.learn(total_timesteps=int(1e5))
    model.save("highway_cnn/model")

    # Record video
    model = DQN.load("highway_cnn/model")

    env = DummyVecEnv([test_env])
    video_length = 2 * env.envs[0].config["duration"]
    env = VecVideoRecorder(env, "highway_cnn/videos/",
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix="dqn-agent")
    obs, info = env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)
    env.close()
