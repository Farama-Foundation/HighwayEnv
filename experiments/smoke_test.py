import gymnasium as gym
import highway_env  # registers environments

env = gym.make("highway-v0", render_mode="rgb_array")  # or "human"
obs, info = env.reset()
print("obs shape:", getattr(obs, "shape", type(obs)))

for _ in range(50):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()

img = env.render()
print("render type:", type(img))
env.close()
