from __future__ import division, print_function
import pygame
from problem import Road, Vehicle, ControlledVehicle, RoadMDP, SimplifiedMDP
import numpy as np
import os

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 30
POLICY_FREQUENCY = 10
dt = 1/FPS

RECORD_VIDEO = False

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
            r.step(dt)

        r.display(sim_surface)
        smdp.display(value_surface)
        screen.blit(sim_surface, (0,0))
        screen.blit(value_surface, (0,SCREEN_HEIGHT/2))
        clock.tick(FPS)
        pygame.display.flip()
        t = t+1

        if RECORD_VIDEO:
            pygame.image.save(screen, "out/highway_{}.bmp".format(t))
            if v.position[0] > np.max([o.position[0] for o in r.vehicles if o is not v])+25:
                os.system("ffmpeg -r 60 -i out/highway_%d.bmp -vcodec libx264 -crf 25 out/highway.avi")
                os.system("rm out/*.bmp")
                done = True

    # Close everything down
    pygame.quit()

if __name__ == '__main__':
    main()