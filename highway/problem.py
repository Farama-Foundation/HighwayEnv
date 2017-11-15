from __future__ import division, print_function
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import random
import pygame
import copy

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

    def __init__(self, position, heading=0, velocity=None, ego=False):
        self.position = np.array(position)
        self.heading = heading
        self.velocity = velocity or 20
        self.ego = ego
        self.color = GREEN if self.ego else YELLOW
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
    def __init__(self, position, heading, velocity, ego, road, target_lane, target_velocity):
        super(ControlledVehicle, self).__init__(position, heading, velocity, ego)
        self.road = road
        self.target_lane = target_lane
        self.target_velocity = target_velocity

    @classmethod
    def create_from(cls, vehicle, road):
        return ControlledVehicle(vehicle.position, vehicle.heading, vehicle.velocity, vehicle.ego, road, road.get_lane(vehicle.position), vehicle.velocity)

    def step(self, dt):
        tau_a = 1.0
        tau_ds = 5.0
        tau_s = 0.7
        Kpa = 1/tau_a
        Kds = 1/(tau_ds*20)
        Kps = 1/tau_s*Kds
        action = {}
        action['steering'] = Kps*(self.road.get_lateral_position(self.target_lane) - self.position[1]) - Kds*self.velocity*np.sin(self.heading)
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
            # self.target_lane += 1
            self.target_lane = self.road.get_lane(self.position)+1
        elif action == "LANE_LEFT":
            # self.target_lane -= 1
            self.target_lane = self.road.get_lane(self.position)-1

        self.target_lane = min(max(self.target_lane, 0), self.road.lanes-1)

    def display(self, screen):
        super(ControlledVehicle, self).display(screen)



class Road(object):
    """
        The set of vehicles on the road, and its characteristics
    """
    STRIPE_SPACING = 5
    STRIPE_LENGTH = 3
    def __init__(self, lanes, lane_width, vehicles=[]):
        self.lanes = lanes
        self.lane_width = lane_width
        self.vehicles = vehicles

    @classmethod
    def create_random_road(cls, lanes, lane_width, vehicles_count=100):
        r = Road(lanes, lane_width)
        for _ in range(vehicles_count):
            r.vehicles.append(r.random_controlled_vehicle())
        return r

    @classmethod
    def create_obstacles_road(cls, lanes, lane_width, rows=4):
        r = Road(lanes, lane_width)
        for d in range(rows):
            for l in range(r.lanes-1):
                v = Vehicle([6*d*r.STRIPE_SPACING, r.get_lateral_position(l + d%2)])
                r.vehicles.append(ControlledVehicle.create_from(v, r))
        return r


    def step(self, dt):
        for vehicle in self.vehicles:
            vehicle.step(dt)

    def get_lane(self, position):
        return int(np.floor(position[1]/self.lane_width))

    def get_lateral_position(self, lane):
        return (lane+0.5)*self.lane_width

    def random_vehicle(self, velocity=None, ego=False):
        l = random.randint(0,self.lanes-1)
        xmin = np.min([v.position[0] for v in self.vehicles]) if len(self.vehicles) else 0
        v = Vehicle([xmin-2*self.STRIPE_SPACING, self.get_lateral_position(l)], 0, velocity, ego)
        return v

    def random_controlled_vehicle(self, velocity=None, ego=False):
        return ControlledVehicle.create_from(self.random_vehicle(velocity, ego), self)

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
    HORIZON = 10.0
    TIME_QUANTIFICATION = 1.0
    ACTION_TIMESTEP = 0.1
    MAX_ACTION_DURATION = 1.0

    def __init__(self, road, ego_vehicle):
        self.road = road
        self.ego_vehicle = ego_vehicle
        self.state = (self.generate_grid(), self.road.get_lane(self.ego_vehicle.position))

    def generate_grid(self):
        grid = np.zeros((self.road.lanes, int(self.HORIZON/self.TIME_QUANTIFICATION)))
        for v in self.road.vehicles:
            if v is not self.ego_vehicle:
                margin = v.LENGTH/2 + self.ego_vehicle.LENGTH/2
                collision_points = [(0, 2), (-margin, 1), (margin, 1)]
                for m, cost in collision_points:
                    distance = v.position[0] - self.ego_vehicle.position[0] + m

                    if self.ego_vehicle.velocity == v.velocity:
                        continue
                    time_of_impact = distance/(self.ego_vehicle.velocity - v.velocity)
                    if time_of_impact < 0:
                        continue
                    l, t = self.road.get_lane(v.position), int(time_of_impact/self.TIME_QUANTIFICATION)
                    if l >= 0 and l < np.shape(grid)[0] and t >= 0 and t < np.shape(grid)[1]:
                        grid[l,t] = max(grid[l,t], cost)
                        # If time of impact is <1 on another lane, a collision will still happen
                        # in a 1s lane chage
                        if t==0:
                            grid[l,1] = max(grid[l,1], 0.5)
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

class SimplifiedMDP(object):
    GAMMA = 0.99
    COLLISION_COST = 1
    LANE_CHANGE_COST = 0.01
    LEFT_LANE_COST = 0.001
    actions = {0:'LANE_LEFT', 1:'IDLE', 2:'LANE_RIGHT'}
    cost = {0:-LANE_CHANGE_COST, 1:0, 2:-LANE_CHANGE_COST}

    def __init__(self, road_state):
        self.grid, self.lane = road_state
        self.value = np.zeros(np.shape(self.grid))
        self.state_reward = -self.COLLISION_COST*self.grid \
                            -self.LEFT_LANE_COST*np.tile(np.arange(np.shape(self.grid)[0])[:, np.newaxis], (1, np.shape(self.grid)[1]))

    def value_iteration(self, steps=20):
        for _ in range(steps):
            self.update()

    def update(self):
        new_value = np.zeros(np.shape(self.value))
        for i in range(np.shape(self.value)[0]):
            for j in range(np.shape(self.value)[1]):
                q_values = self.get_q_values(i,j)
                if len(q_values):
                    new_value[i,j] = self.GAMMA*np.max(q_values)
                else:
                    new_value[i,j] = self.state_reward[i,j]
        self.value = new_value

    def clamp_position(self, i, j):
        return min(max(i, 0), np.shape(self.value)[0]-1),  min(max(j, 0), np.shape(self.value)[1]-1)

    def reward(self, i, j, action):
        return self.state_reward[i,j]+self.cost[action]

    def get_q_values(self, i, j):
        q_values = []
        for k in range(0,3):
            p, q = self.clamp_position(i+k-1,j+1)
            q_values.append(self.reward(i,j,k)+self.value[p,q])
        return q_values

    def pick_action(self):
        i, j = self.lane, 0
        q_values = self.get_q_values(i, j)
        a = np.argmax(q_values)
        return self.actions[a]

    def display(self, surface):
        norm = mpl.colors.Normalize(vmin=-3, vmax=1)
        cmap = cm.jet_r
        cell_size = (surface.get_width()//np.shape(self.value)[1], surface.get_height()//np.shape(self.value)[0])
        for i in range(np.shape(self.value)[0]):
            for j in range(np.shape(self.value)[1]):
                color = cmap(norm(self.value[i,j]), bytes=True)
                pygame.draw.rect(surface, color, (j*cell_size[0],i*cell_size[1],cell_size[0],cell_size[1]), 0)


# class Qlearning(object):
#     def __init__(self, mdp):
#         self.mdp = mdp
#         self.Q = {}

#     def update(self):
#         for a in ['IDLE', 'LANE_LEFT', 'LANE_RIGHT']:
#             state_action_hash = mdp.state[0].tostring()+mdp.state[1]+a
#             print(state_action_hash)


def test():
    r = Road.create_random_road(4, 4.0, vehicles_count=1)
    v = Vehicle([-20, r.get_lateral_position(0)], 0, 25, ego=True)
    v = ControlledVehicle.create_from(v, r)
    r.vehicles.append(v)

    for _ in range(5):
        mdp = RoadMDP(r, v)
        smdp = SimplifiedMDP(mdp.state)
        smdp.value_iteration()
        action = smdp.pick_action()
        print(mdp.state)
        print(smdp.value)
        print(action)
        mdp.step(action)

if __name__ == '__main__':
    test()