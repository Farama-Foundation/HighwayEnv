import numpy as np
import pygame

from highway_env.envs.random_road_env import RandomRoadEnv


def test_random_road_env():
    env = RandomRoadEnv(render_mode="human")

    pygame.init()
    running = True

    while running:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            if info["parked"]:
                print("Parked")
            if info["crashed"]:
                print("Crashed")
            break

    pygame.quit()
    env.close()
