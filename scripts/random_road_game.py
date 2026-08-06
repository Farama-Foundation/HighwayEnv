import gymnasium as gym
import numpy as np
import pygame

import highway_env


generation_params = {
    "target_num_endpoints": 200,
    "disable_prints": False,
}


config = {
    "route_following_reward_scalar": 0,
    "generation_params": generation_params,
    "parking_seed": None,
}

gym.register_envs(highway_env)
env = gym.make("random-road-v0", render_mode="human")
env.reset(seed=None, options={"config": config})

throttle_speed = 0.01
steer_speed = 0.5
rolling_friction = 0.001  # rolling friction + air resistance + engine braking
break_multiplier = 3


pygame.init()
clock = pygame.time.Clock()
FPS = 15
print("WASD to drive; O/L to zoom in/out.")

env.unwrapped.viewer.sim_surface.scaling = 8

cumulative_return = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

    if not running:
        break

    keys = pygame.key.get_pressed()

    # Car control #
    throttle = 0
    steer = 0
    if keys[pygame.K_w]:
        throttle += throttle_speed
    if keys[pygame.K_s]:
        throttle -= throttle_speed
    if keys[pygame.K_a]:
        steer -= steer_speed
    if keys[pygame.K_d]:
        steer += steer_speed

    if throttle * env.unwrapped.vehicle.speed < 0:  # Simulating 'breaking'
        throttle *= break_multiplier

    acceleration = throttle - rolling_friction * env.unwrapped.vehicle.speed

    action = np.array([acceleration, steer], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)
    cumulative_return += reward
    if terminated or truncated:
        if info["parked"]:
            print("Parked")
        if info["crashed"]:
            print("Crashed")

        break

    if not pygame.display.get_init():
        break

    clock.tick(FPS)


print("Cumulated return:", cumulative_return)


pygame.quit()
env.close()
