import sys

import numpy as np
import pygame

import gymnasium as gym
import highway_env
from highway_env.road.generation.generator import (
    default_params,
    load_lanes_from_disk,
    save_lanes_to_disk,
)


# RandomRoadEnv creation #
params = default_params()
# params['seed'] = 0

lanes = None
if len(sys.argv) > 1:
    print("Loading road network...")
    lanes = load_lanes_from_disk(sys.argv[1])

gym.register_envs(highway_env)
env = gym.make("random-road-v0", render_mode="human", lanes=lanes, generation_params=params)
env = env.unwrapped

if len(sys.argv) <= 1:
    save_lanes_to_disk("lanes_saved/lanes_saved.npz", env.lanes)

env.config["route_following_reward_scalar"] = 0

# Car constants #
throttle_speed = 0.01
steer_speed = 0.5
rolling_friction = 0.001  # rolling friction + air resistance + engine braking
break_multiplier = 3

# Render loop #
env.viewer.sim_surface.scaling = 8
pygame.init()
clock = pygame.time.Clock()
FPS = 15
running = True
cumulative_return = 0


print("WASD to drive; O/L to zoom in/out.")
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

    if throttle * env.vehicle.speed < 0:  # Simulating 'breaking'
        throttle *= break_multiplier

    acceleration = throttle - rolling_friction * env.vehicle.speed

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
