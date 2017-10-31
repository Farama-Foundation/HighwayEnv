from __future__ import division, print_function
import pygame
from problem import Road, Vehicle
import numpy as np
import random
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 200
FPS = 30
dt = 1/FPS

def main():
    r = Road(4, 4.0, [])

    pygame.init()
    done = False
    size = [SCREEN_WIDTH, SCREEN_HEIGHT]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Highway")
    clock = pygame.time.Clock()
    v = Vehicle(np.array([0, 0]), 0, 0)
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    v.action['acceleration'] = 3
                if event.key == pygame.K_LEFT:
                    v.action['acceleration'] = -3
                if event.key == pygame.K_DOWN:
                    v.action['steering'] = 4*np.pi/180
                if event.key == pygame.K_UP:
                    v.action['steering'] = -4*np.pi/180
                if event.key == pygame.K_SPACE:
                    v = Vehicle(np.array([0, 2.0+random.randint(0,r.lanes-1)*r.lane_width]), 0, 20)
                    r.vehicles.append(v)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_RIGHT:
                    v.action['acceleration'] = 0
                if event.key == pygame.K_LEFT:
                    v.action['acceleration'] = 0
                if event.key == pygame.K_DOWN:
                    v.action['steering'] = 0
                if event.key == pygame.K_UP:
                    v.action['steering'] = 0

        r.step(dt)
        print(r)
        r.display(screen)
        clock.tick(FPS)

        pygame.display.flip()

    # Close everything down
    pygame.quit()

if __name__ == '__main__':
    main()