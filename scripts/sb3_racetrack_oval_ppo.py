# train and test PPO agent on racetrack-oval-v0 with config

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env  # noqa: F401


TRAIN = True

# env configuration
config = {
    "observation": {
        "type": "OccupancyGrid",
        "features": [
            "presence",
            "on_road",
            "x",
            "y",
            "vx",
            "vy",
            "cos_h",
            "sin_h",
            "long_off",
            "lat_off",
            "ang_off",
        ],
        "grid_size": [[-18, 18], [-18, 18]],
        "grid_step": [3, 3],
        "as_image": False,
        "align_to_vehicle_axes": True,
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": False,
        "lateral": True,
        "target_speeds": [0, 5, 10],
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 120,
    "collision_reward": -1000,
    "lane_centering_cost": 4,
    "lane_centering_reward": 1,
    "action_reward": -100,
    "controlled_vehicles": 1,
    "other_vehicles": 3,
    "screen_width": 1000,
    "screen_height": 1000,
    "centering_position": [0.5, 0.5],
    "speed_limit": 10.0,
    "terminate_off_road": True,  # CL: terminate if car goes off-road
    "length": 100,  # CL: length of straight; 0: random number form [100,200]
    "no_lanes": 3,  # CL: no. of lanes; 0: random number form [2,7]
}


if __name__ == "__main__":
    n_cpu = 8
    batch_size = 32

    def make_env():
        return gym.make("racetrack-oval-v0", config=config)

    env = make_vec_env(make_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        verbose=2,
        tensorboard_log="racetrack_oval_ppo/",
    )
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e3))
        model.save("racetrack_oval_ppo/model")
        del model

    # Run the algorithm
    model = PPO.load("racetrack_oval_ppo/model", env=env)

    env = gym.make("racetrack-oval-v0", render_mode="rgb_array", config=config)
    env = RecordVideo(
        env, video_folder="racetrack_oval_ppo/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)

    for video in range(5):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
