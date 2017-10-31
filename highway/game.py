from __future__ import division, print_function
import pygame
from problem import Road, ControlledVehicle
import numpy as np
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 200
FPS = 30
dt = 1/FPS

def main():
    r = Road(4, 4.0, [])
    for _ in range(10):
        r.vehicles.append(r.random_controller())
    v = r.random_controller()
    r.vehicles.append(v)

    pygame.init()
    done = False
    size = [SCREEN_WIDTH, SCREEN_HEIGHT]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Highway")
    clock = pygame.time.Clock()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            v.handle_event(event)

        r.step(dt)
        print(v)
        r.display(screen)
        clock.tick(FPS)

        pygame.display.flip()

    # Close everything down
    pygame.quit()

if __name__ == '__main__':
    main()