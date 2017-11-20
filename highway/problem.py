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
        s = pygame.Surface((pix(self.LENGTH), pix(self.WIDTH)), pygame.SRCALPHA)   # per-pixel alpha
        s.fill(self.color)
        pygame.draw.rect(s, BLACK, (0,0,pix(self.LENGTH),pix(self.WIDTH)), 1)
        s = s.convert_alpha()
        h = self.heading if abs(self.heading) > 2*np.pi/180 else 0
        sr = pygame.transform.rotate(s, -h*180/np.pi)
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

    SPEED_MIN = 20
    SPEED_COUNT = 3
    SPEED_MAX = 35

    def __init__(self, position, heading, velocity, ego, road, target_lane, target_velocity):
        super(ControlledVehicle, self).__init__(position, heading, velocity, ego)
        self.road = road
        self.target_lane = target_lane
        self.target_velocity = self.speed_to_index(target_velocity)

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
        action['acceleration'] = Kpa*(self.index_to_speed(self.target_velocity) - self.velocity)

        super(ControlledVehicle, self).step(dt, action)

    @classmethod
    def index_to_speed(cls, index):
        return cls.SPEED_MIN+index*(cls.SPEED_MAX-cls.SPEED_MIN)/(cls.SPEED_COUNT-1)

    @classmethod
    def speed_to_index(cls, speed):
        x = (speed - cls.SPEED_MIN)/(cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.round(x*(cls.SPEED_COUNT-1)))

    def speed_index(self):
        return self.speed_to_index(self.velocity)

    def get_lane(self):
        return self.road.get_lane(self.position)

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
            self.target_velocity = self.speed_to_index(self.velocity) + 1
        elif action == "SLOWER":
            self.target_velocity = self.speed_to_index(self.velocity) - 1
        elif action == "LANE_RIGHT":
            self.target_lane = self.get_lane()+1
        elif action == "LANE_LEFT":
            self.target_lane = self.get_lane()-1

        self.target_velocity = min(max(self.target_velocity, 0), self.SPEED_COUNT-1)
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
        for d in range(1,rows+1):
            for l in range(r.lanes-1):
                v = Vehicle([5*d*r.STRIPE_SPACING, r.get_lateral_position(l + d%2)])
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

        self.T = int(self.HORIZON/self.TIME_QUANTIFICATION)
        self.V = self.ego_vehicle.SPEED_COUNT

        self.state = self.get_state()


    def get_state(self):
        grids = np.zeros((self.V, self.road.lanes, self.T))
        for k in range(self.V):
            grids[k,:,:] = self.generate_2D_grid(self.ego_vehicle.index_to_speed(k))
        lane = self.road.get_lane(self.ego_vehicle.position)
        speed = self.ego_vehicle.speed_index()
        return grids, lane, speed

    def generate_2D_grid(self, ego_velocity):
        grid = np.zeros((self.road.lanes, self.T))
        for v in self.road.vehicles:
            if v is not self.ego_vehicle:
                margin = v.LENGTH/2 + self.ego_vehicle.LENGTH/2
                collision_points = [(0, 2), (-margin, 1), (margin, 1)]
                for m, cost in collision_points:
                    distance = v.position[0] - self.ego_vehicle.position[0] + m

                    if ego_velocity == v.velocity:
                        continue
                    time_of_impact = distance/(ego_velocity - v.velocity)
                    if time_of_impact < 0:
                        continue
                    l, t = self.road.get_lane(v.position), int(time_of_impact/self.TIME_QUANTIFICATION)
                    if l >= 0 and l < np.shape(grid)[0] and t >= 0 and t < np.shape(grid)[1]:
                        grid[l,t] = max(grid[l,t], cost)
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
    GAMMA = 1.0
    COLLISION_COST = 10
    LANE_CHANGE_COST = 0.00
    LEFT_LANE_COST = 0.00
    HIGH_VELOCITY_REWARD = 0.5
    actions = {0:'IDLE', 1:'LANE_LEFT', 2:'LANE_RIGHT', 3:'FASTER', 4:'SLOWER'}
    cost = {0:0, 1:-LANE_CHANGE_COST, 2:-LANE_CHANGE_COST, 3:0, 4:0}

    def __init__(self, road_state):
        self.grids, self.lane, self.speed = road_state

        self.V, self.L, self.T = np.shape(self.grids)
        self.value = np.zeros(np.shape(self.grids))
        self.state_reward = -self.COLLISION_COST*self.grids \
                            -self.LEFT_LANE_COST*np.tile(np.arange(self.L)[np.newaxis, :, np.newaxis], (self.V, 1, self.T)) \
                            +self.HIGH_VELOCITY_REWARD*np.tile(np.arange(self.V)[:, np.newaxis, np.newaxis], (1, self.L, self.T))

    def value_iteration(self, steps=50):
        for _ in range(steps):
            self.update()

    def update(self):
        new_value = np.zeros(np.shape(self.value))
        for h in range(self.V):
            for i in range(self.L):
                for j in range(self.T):
                    q_values = self.get_q_values(h,i,j)
                    if len(q_values):
                        new_value[h,i,j] = self.GAMMA*np.max(q_values)
                    else:
                        new_value[h,i,j] = self.state_reward[h,i,j]
        self.value = new_value

    def clamp_position(self, h, i, j):
        o = min(max(h, 0), np.shape(self.value)[0]-1)
        p = min(max(i, 0), np.shape(self.value)[1]-1)
        q = min(max(j, 0), np.shape(self.value)[2]-1)
        return o, p, q

    def reward(self, h, i, j, action):
        return self.state_reward[h,i,j] + self.cost[action]

    def transition_model(self, k, h, i, j):
        """
            Deterministric transition from a position in the grid to the next.
            k: action index
            h: velocity index
            i: lane index
            j: time index
        """
        if k == 0:
            return self.clamp_position(h,i,j+1) # IDLE
        elif k == 1:
            return self.clamp_position(h,i-1,j+1) # LEFT
        elif k == 2:
            return self.clamp_position(h,i+1,j+1) # RIGHT
        elif k == 3:
            return self.clamp_position(h+1,i,j+1) # FASTER
        elif k == 4:
            return self.clamp_position(h-1,i,j+1) # SLOWER
        else:
            return None

    def get_q_values(self, h, i, j):
        q_values = []
        if j == self.T - 1:
            return q_values # Terminal state
        for k in range(0,5):
            o, p, q = self.transition_model(k,h,i,j)
            q_values.append(self.reward(h,i,j,k)+self.value[o,p,q])
        return q_values

    def pick_action(self):
        h, i, j = self.speed, self.lane, 0
        q_values = self.get_q_values(h, i, j)
        a = np.argmax(q_values)
        return self.actions[a]

    def display(self, surface):
        norm = mpl.colors.Normalize(vmin=-30, vmax=20)
        cmap = cm.jet_r
        cell_size = (surface.get_width()//self.T, surface.get_height()//(self.L*self.V))
        velocity_size = surface.get_height()//self.V
        for h in range(self.V):
                for i in range(self.L):
                    for j in range(self.T):
                        color = cmap(norm(self.value[h,i,j]), bytes=True)
                        pygame.draw.rect(surface, color, (j*cell_size[0],i*cell_size[1]+h*velocity_size,cell_size[0],cell_size[1]), 0)
        pygame.draw.rect(surface, (255,0,0), (0*cell_size[0],self.lane*cell_size[1]+self.speed*velocity_size,cell_size[0],cell_size[1]), 1)

def test():
    r = Road.create_random_road(4, 4.0, vehicles_count=1)
    v = Vehicle([-20, r.get_lateral_position(0)], 0, 25, ego=True)
    v = ControlledVehicle.create_from(v, r)
    r.vehicles.append(v)


    mdp = RoadMDP(r, v)
    print(mdp.state)
    smdp = SimplifiedMDP(mdp.state)
    print(smdp.value)
    smdp.value_iteration()
    print(smdp.value)
    # action = smdp.pick_action()
    # print(action)
    # mdp.step(action)

if __name__ == '__main__':
    test()