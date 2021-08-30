import gym
from stable_baselines3 import DQN

import highway_env


if __name__ == '__main__':
    # Create the environment
    env = gym.make("highway-fast-v0")
    obs = env.reset()

    # Create the model
    model = DQN('MlpPolicy', env,
                gamma=0.8,
                learning_rate=5e-4,
                buffer_size=40*1000,
                learning_starts=200,
                exploration_fraction=0.6,
                batch_size=128,
                verbose=1,
                tensorboard_log="highway_dqn/")

    # Train the model
    model.learn(total_timesteps=int(1e5))
    model.save("dqn_highway")

    # Run the algorithm
    model.load("dqn_highway")
    for _ in range(10):
        done = False
        obs = env.reset()
        while not done:
            # Predict
            print(obs)
            action, _states = model.predict(obs)
            # Get reward
            obs, reward, done, info = env.step(action)
            # Render
            env.render()

    env.close()
