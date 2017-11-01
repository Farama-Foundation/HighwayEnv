from __future__ import division, print_function
import pygame
from problem import Road, Vehicle, ControlledVehicle, RoadMDP, SimplifiedMDP
import numpy as np
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 200
FPS = 30
dt = 1/FPS

def main():
    r = Road(4, 4.0, [])
    for _ in range(100):
        r.vehicles.append(r.random_controller())
    v = r.random_controller(25, ego=True)
    # r.generate_controller_rows()
    # v = Vehicle([-30, r.get_lateral_position(0)], 0, 25)
    v = ControlledVehicle(v.position, v.heading, v.velocity, r, r.get_lane(v.position), v.velocity, ego=True)
    r.vehicles.append(v)

    t = 0
    pygame.init()
    done = False
    pause = False
    size = [SCREEN_WIDTH, SCREEN_HEIGHT]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Highway")
    clock = pygame.time.Clock()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause = not pause
            v.handle_event(event)

        if t % 3 == 0:
            mdp = RoadMDP(r, v)
            smdp = SimplifiedMDP(mdp.state)
            smdp.value_iteration()
            print(mdp.state)
            print(smdp.value)
            action = smdp.pick_action()
            print(action)
            v.perform_action(action)
            # pause = True

        if not pause:
            r.step(dt)
        r.display(screen)
        clock.tick(FPS)
        pygame.display.flip()
        t = t+1

        # print(mdp.generate_grid())

    # Close everything down
    pygame.quit()

if __name__ == '__main__':
    main()