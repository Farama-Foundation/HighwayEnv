import pytest
import timeit
import gymnasium as gym
import highway_env

highway_env.register_highway_envs()


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def time_env(env_name, steps=20):
    env = gym.make(env_name)
    env.reset()
    for _ in range(steps):
        _, _, done, truncated, _ = env.step(env.action_space.sample())
        env.reset() if done or truncated else _
    env.close()


def test_running_time(repeat=1):
    for env_name, steps in [
        ("highway-v0", 10),
        ("highway-fast-v0", 10),
        ("parking-v0", 20)
    ]:
        env_time = wrapper(time_env, env_name, steps)
        time_spent = timeit.timeit(env_time, number=repeat) / repeat
        env = gym.make(env_name)
        time_simulated = steps / env.unwrapped.config["policy_frequency"]
        real_time_ratio = time_simulated / time_spent
        print("Real time ratio for {}: {}".format(env_name, real_time_ratio))
        assert real_time_ratio > 0.5  # let's not be too ambitious for now


if __name__ == "__main__":
    test_running_time()
