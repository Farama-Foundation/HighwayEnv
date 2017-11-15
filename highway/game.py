from __future__ import division, print_function
import pygame
from problem import Road, Vehicle, ControlledVehicle, RoadMDP, SimplifiedMDP
import numpy as np
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 30
POLICY_FREQUENCY = 10
dt = 1/FPS

def main():
    r = Road.create_random_road(4, 4.0, 100)
    # r = Road.create_obstacles_road(4, 4.0)
    v = r.random_controlled_vehicle(25, ego=True)
    # v = Vehicle([-20, r.get_lateral_position(0)], 0, 25, ego=True)
    # v = ControlledVehicle.create_from(v, r)
    r.vehicles.append(v)

    t = 0
    done = False
    pause = False
    pygame.init()
    pygame.display.set_caption("Highway")
    clock = pygame.time.Clock()

    size = [SCREEN_WIDTH, SCREEN_HEIGHT]
    screen = pygame.display.set_mode(size)
    sim_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT/2))
    value_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT/2))

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause = not pause
            v.handle_event(event)

        if not pause:
            if t % (FPS//POLICY_FREQUENCY) == 0:
                mdp = RoadMDP(r, v)
                smdp = SimplifiedMDP(mdp.state)
                smdp.value_iteration()
                print(mdp.state)
                print(smdp.value)
                action = smdp.pick_action()
                print(action)
                v.perform_action(action)
                # pause = True
            r.step(dt)

        r.display(sim_surface)
        smdp.display(value_surface)
        screen.blit(sim_surface, (0,0))
        screen.blit(value_surface, (0,SCREEN_HEIGHT/2))
        clock.tick(FPS)
        pygame.display.flip()
        t = t+1

        # print(mdp.generate_grid())

    # Close everything down
    pygame.quit()

if __name__ == '__main__':
    main()