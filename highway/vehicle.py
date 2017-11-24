from __future__ import division, print_function
import numpy as np
import pygame

class Vehicle(object):
    """
        A moving vehicle and its dynamics.
    """
    LENGTH = 5.0
    WIDTH = 2.0

    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)

    def __init__(self, position, heading=0, velocity=None, ego=False):
        self.position = np.array(position)
        self.heading = heading
        self.velocity = velocity or 20
        self.ego = ego
        self.color = self.GREEN if self.ego else self.YELLOW
        self.action = {'steering':0, 'acceleration':0}

    def step(self, dt, action=None):
        if not action:
            action = self.action
        v = self.velocity*np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v*dt
        self.heading += self.velocity*np.tan(action['steering'])/self.LENGTH*dt
        self.velocity += action['acceleration']*dt

    def handle_event(self, event):
        if not self.action:
            self.action = {}
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                self.action['acceleration'] = 3
            if event.key == pygame.K_LEFT:
                self.action['acceleration'] = -3
            if event.key == pygame.K_DOWN:
                self.action['steering'] = 20*np.pi/180
            if event.key == pygame.K_UP:
                self.action['steering'] = -20*np.pi/180
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT:
                self.action['acceleration'] = 0
            if event.key == pygame.K_LEFT:
                self.action['acceleration'] = 0
            if event.key == pygame.K_DOWN:
                self.action['steering'] = 0
            if event.key == pygame.K_UP:
                self.action['steering'] = 0

    def display(self, screen):
        s = pygame.Surface((screen.pix(self.LENGTH), screen.pix(self.WIDTH)), pygame.SRCALPHA)   # per-pixel alpha
        s.fill(self.color)
        pygame.draw.rect(s, screen.BLACK, (0,0,screen.pix(self.LENGTH),screen.pix(self.WIDTH)), 1)
        s = s.convert_alpha()
        h = self.heading if abs(self.heading) > 2*np.pi/180 else 0
        sr = pygame.transform.rotate(s, -h*180/np.pi)
        screen.blit(sr, (screen.pos2pix(self.position[0]-self.LENGTH/2, self.position[1]-self.WIDTH/2)))

    def __str__(self):
        return "{}, {}, {}".format(self.position, self.heading, self.velocity)

    def __repr__(self):
        return self.__str__()

class ControlledVehicle(Vehicle):
    """
        A vehicle piloted by a low-level controller, allowing high-level actions
        such as lane changes.
    """

    def __init__(self, position, heading, velocity, ego, road, target_lane, target_velocity):
        super(ControlledVehicle, self).__init__(position, heading, velocity, ego)
        self.road = road
        self.target_lane = target_lane
        self.target_velocity = target_velocity

    @classmethod
    def create_from(cls, vehicle, road):
        return ControlledVehicle(vehicle.position, vehicle.heading, vehicle.velocity, vehicle.ego, road, road.get_lane(vehicle.position), vehicle.velocity)

    def step(self, dt):
        tau_a = 0.1
        tau_ds = 5.0
        tau_s = 0.4
        Kpa = 1/tau_a
        Kds = 1/(tau_ds*5)
        Kps = 1/tau_s*Kds
        action = {}
        action['steering'] = Kps*(self.road.get_lateral_position(self.target_lane) - self.position[1]) - Kds*self.velocity*np.sin(self.heading)
        action['acceleration'] = Kpa*(self.target_velocity - self.velocity)

        super(ControlledVehicle, self).step(dt, action)

    def get_lane(self):
        return self.road.get_lane(self.position)

    def display(self, screen):
        super(ControlledVehicle, self).display(screen)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                self.perform_action("FASTER")
            if event.key == pygame.K_LEFT:
                self.perform_action("SLOWER")
            if event.key == pygame.K_DOWN:
                self.perform_action("LANE_RIGHT")
            if event.key == pygame.K_UP:
                self.perform_action("LANE_LEFT")

    def perform_action(self, action):
        if action == "FASTER":
            self.target_velocity += 5
        elif action == "SLOWER":
            self.target_velocity -= 5
        elif action == "LANE_RIGHT":
            self.target_lane += 1
        elif action == "LANE_LEFT":
            self.target_lane -= 1


class MDPVehicle(ControlledVehicle):
    """
        A vehicle piloted by a low-level controller, allowing high-level actions
        such as lane changes.
    """

    SPEED_MIN = 21
    SPEED_COUNT = 3
    SPEED_MAX = 35

    def __init__(self, position, heading, velocity, ego, road, target_lane, target_velocity):
        super(MDPVehicle, self).__init__(position, heading, velocity, ego, road, target_lane, target_velocity)
        self.velocity_index = self.speed_to_index(target_velocity)
        self.target_velocity = self.index_to_speed(self.velocity_index)

    @classmethod
    def create_from(cls, vehicle, road):
        return MDPVehicle(vehicle.position, vehicle.heading, vehicle.velocity, vehicle.ego, road, road.get_lane(vehicle.position), vehicle.velocity)

    def step(self, dt):
        self.target_velocity = self.index_to_speed(self.velocity_index)
        super(MDPVehicle, self).step(dt)


    @classmethod
    def index_to_speed(cls, index):
        return cls.SPEED_MIN+index*(cls.SPEED_MAX-cls.SPEED_MIN)/(cls.SPEED_COUNT-1)

    @classmethod
    def speed_to_index(cls, speed):
        x = (speed - cls.SPEED_MIN)/(cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.round(x*(cls.SPEED_COUNT-1)))

    def speed_index(self):
        return self.speed_to_index(self.velocity)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                self.perform_action("FASTER")
            if event.key == pygame.K_LEFT:
                self.perform_action("SLOWER")
            if event.key == pygame.K_DOWN:
                self.perform_action("LANE_RIGHT")
            if event.key == pygame.K_UP:
                self.perform_action("LANE_LEFT")

    def perform_action(self, action):
        if action == "FASTER":
            self.velocity_index = self.speed_to_index(self.velocity) + 1
        elif action == "SLOWER":
            self.velocity_index = self.speed_to_index(self.velocity) - 1
        elif action == "LANE_RIGHT":
            self.target_lane = self.get_lane()+1
        elif action == "LANE_LEFT":
            self.target_lane = self.get_lane()-1

        self.velocity_index = min(max(self.velocity_index, 0), self.SPEED_COUNT-1)
        self.target_lane = min(max(self.target_lane, 0), self.road.lanes-1)


    def display(self, screen):
        super(ControlledVehicle, self).display(screen)

def test():
    v = Vehicle([-20., 1.], 0, 20, ego=True)

    print(v)
    for _ in range(10):
        v.step(0.1)
    print(v)

if __name__ == '__main__':
    test()