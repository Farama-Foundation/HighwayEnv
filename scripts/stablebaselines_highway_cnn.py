import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder


if __name__ == '__main__':
    # Train
    env = gym.make('highway-v0')
    env.configure({
        "lanes_count": 3,
        "vehicles_count": 15,
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
        "policy_frequency": 2,
        "duration": 40,
    })
    env.reset()
    model = DQN('CnnPolicy', env,
                gamma=0.8,
                learning_rate=5e-4,
                buffer_size=40*1000,
                learning_starts=200,
                exploration_fraction=0.6,
                target_update_interval=256,
                batch_size=32,
                verbose=1,
                tensorboard_log="logs/")
    model.learn(total_timesteps=int(2e5))
    model.save("dqn_highway")

    # Record video
    model = DQN.load("dqn_highway")
    env.configure({"policy_frequency": 15, "duration": 20 * 15})
    video_length = 2 * env.config["duration"]
    env = VecVideoRecorder(env, "videos/",
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix="dqn-agent")
    obs = env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
    env.close()
