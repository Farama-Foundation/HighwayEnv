from __future__ import division, print_function
import numpy as np
import random
import pygame

BLACK = (0, 0, 0)
GREY = (100, 100, 100)
GREEN = (50, 200, 0)
YELLOW = (200, 200, 0)
WHITE = (255, 255, 255)
SCALING = 10.0

origin = [0, 0]

def pix(length):
    return int(length*SCALING)

def pos2pix(x, y):
    global origin
    return pix(x-origin[0]), pix(y-origin[1])

class Vehicle(object):
    """
        A moving vehicle and its dynamics.
    """
    LENGTH = 5.0
    WIDTH = 2.0

    def __init__(self, position, heading, velocity, ego=False):
        self.position = np.array(position)
        self.heading = heading
        self.velocity = velocity
        self.color = GREEN if ego else YELLOW
        self.action = {'steering':0, 'acceleration':0}

    def step(self, dt, action=None):
        if not action:
            action = self.action
        v = self.velocity*np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v*dt
        self.heading += self.velocity*action['steering']*dt
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
                self.action['steering'] = 4*np.pi/180
            if event.key == pygame.K_UP:
                self.action['steering'] = -4*np.pi/180
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
        s = pygame.Surface((pix(self.LENGTH), pix(self.WIDTH)), pygame.SRCALPHA)   # per-pixel alpha
        s.fill(self.color)
        pygame.draw.rect(s, BLACK, (0,0,pix(self.LENGTH),pix(self.WIDTH)), 1)
        s = s.convert_alpha()
        sr = pygame.transform.rotate(s, -self.heading*180/np.pi)
        screen.blit(sr, (pos2pix(self.position[0]-self.LENGTH/2, self.position[1]-self.WIDTH/2)))

    def __str__(self):
        return "{}, {}, {}".format(self.position, self.heading, self.velocity)

    def __repr__(self):
        return self.__str__()

class ControlledVehicle(Vehicle):
    """
        A vehicle piloted by a low-level controller, allowing high-level actions
        such as lane changes.
    """
    def __init__(self, position, heading, velocity, road, target_lane, target_velocity, ego=False):
        super(ControlledVehicle, self).__init__(position, heading, velocity, ego)
        self.road = road
        self.target_lane = target_lane
        self.target_velocity = target_velocity

    def step(self, dt):
        tau_a = 1.0
        tau_ds = 5.0
        tau_s = 0.7
        Kpa = 1/tau_a
        Kds = 1/(tau_ds*20)
        Kps = 1/tau_s*Kds
        action = {}
        action['steering'] = Kps*((self.target_lane+0.5)*self.road.lane_width - self.position[1]) - Kds*self.velocity*np.sin(self.heading)
        action['acceleration'] = Kpa*(self.target_velocity - self.velocity)
        # action = None

        super(ControlledVehicle, self).step(dt, action)

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

    def display(self, screen):
        super(ControlledVehicle, self).display(screen)



class Road(object):
    """
        The set of vehicles on the road, and its characteristics
    """
    STRIPE_SPACING = 5
    STRIPE_LENGTH = 3
    def __init__(self, lanes, lane_width, vehicles=None):
        self.lanes = lanes
        self.lane_width = lane_width
        self.vehicles = vehicles

    def step(self, dt):
        for vehicle in self.vehicles:
            vehicle.step(dt)

    def get_lane(self, position):
        return int(np.floor(position[1]/self.lane_width))

    def random_vehicle(self, velocity=None, ego=False):
        l = random.randint(0,self.lanes-1)
        velocity = velocity or 20
        v = Vehicle([-2*self.STRIPE_SPACING*len(self.vehicles), (l+0.5)*self.lane_width], 0, velocity, ego)
        return v

    def random_controller(self, velocity=None, ego=False):
        v = self.random_vehicle(velocity)
        return ControlledVehicle(v.position, v.heading, v.velocity, self, self.get_lane(v.position), v.velocity, ego)

    def display(self, screen):
        screen.fill(GREY)
        global origin
        if len(self.vehicles):
            origin = self.vehicles[-1].position-np.array([3*self.STRIPE_SPACING, screen.get_height()/(2*SCALING)])

        # Draw tracks
        x0 = (int(origin[0])//self.STRIPE_SPACING)*self.STRIPE_SPACING
        ticks = int(screen.get_width()/(self.STRIPE_SPACING*SCALING))+1
        # Outer
        pygame.draw.line(screen, WHITE,
            (pos2pix(x0 + 0*self.STRIPE_SPACING, 0*self.lane_width)),
            (pos2pix(x0 + ticks*self.STRIPE_SPACING + self.STRIPE_LENGTH, 0*self.lane_width)), 2)
        pygame.draw.line(screen, WHITE,
            (pos2pix(x0 + 0*self.STRIPE_SPACING, self.lanes*self.lane_width)),
            (pos2pix(x0 + ticks*self.STRIPE_SPACING + self.STRIPE_LENGTH, self.lanes*self.lane_width)), 2)
        # Inner
        for l in range(1,self.lanes):
            for k in range(ticks):
                pygame.draw.line(screen, WHITE,
                    (pos2pix(x0 + k*self.STRIPE_SPACING, l*self.lane_width)),
                    (pos2pix(x0 + k*self.STRIPE_SPACING + self.STRIPE_LENGTH, l*self.lane_width)), 2)

        for v in self.vehicles:
            v.display(screen)

    def __repr__(self):
        return self.vehicles.__repr__()

class RoadMDP(object):
    """
        A MDP representing the times to collision between the ego-vehicle and
        other vehicles on the road.
    """
    TIME_QUANTIFICATION = 1.0
    ACTION_TIMESTEP = 0.1
    MAX_ACTION_DURATION = 1.0

    def __init__(self, road, ego_vehicle):
        self.road = road
        self.ego_vehicle = ego_vehicle
        self.state = (self.generate_grid(), self.road.get_lane(self.ego_vehicle.position))

    def generate_grid(self):
        grid = np.zeros((self.road.lanes, 10))
        for v in self.road.vehicles:
            if v is not self.ego_vehicle:
                distance = v.position[0] - self.ego_vehicle.position[0] - v.LENGTH/2 - self.ego_vehicle.LENGTH/2
                if distance < 0 or self.ego_vehicle.velocity == v.velocity:
                    continue
                time_of_impact = distance/(self.ego_vehicle.velocity - v.velocity)
                l, t = self.road.get_lane(v.position), int(time_of_impact/self.TIME_QUANTIFICATION)
                if l >= 0 and l < np.shape(grid)[0] and t >= 0 and t < np.shape(grid)[1]:
                    grid[l,t] = 1
        return grid

    def step(self, action):
        # Send order to low-level agent
        self.ego_vehicle.perform_action(action)

        # Inner high-frequency loop
        for k in range(int(self.MAX_ACTION_DURATION/self.ACTION_TIMESTEP)):
            self.road.step(self.ACTION_TIMESTEP)
            new_state = (self.generate_grid(), self.road.get_lane(self.ego_vehicle.position))
            # Stop whenever macro-state changes
            # if (self.state[0] != new_state[0]).any() or self.state[1] != new_state[1]:
            #     break
        self.state = new_state
        return self.state


def test():
    r = Road(4, 4.0, [])
    for _ in range(10):
        r.vehicles.append(r.random_controller())
    v = r.random_controller(25, ego=True)
    r.vehicles.append(v)

    mdp = RoadMDP(r, v)
    print(mdp.step('IDLE'))
    print(mdp.step('IDLE'))
    print(mdp.step('LANE_LEFT'))
    print(mdp.step('IDLE'))

if __name__ == '__main__':
    test()