from __future__ import division, print_function
import numpy as np
import pygame
import random
import copy

def wrap_to_pi(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi

class Vehicle(object):
    """
        A moving vehicle and its dynamics.
    """
    LENGTH = 5.0
    WIDTH = 2.0

    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    BLUE = (100, 200, 255)

    id_max = 0

    def __init__(self, position, heading=0, velocity=None, ego=False):
        self.position = np.array(position)
        self.heading = heading
        self.velocity = velocity or 20 - random.randint(0,3)
        self.ego = ego
        self.color = self.GREEN if self.ego else self.YELLOW
        self.action = {'steering':0, 'acceleration':0}

        self.lane = None
        self.id = self.id_max
        self.id_max += 1

    @classmethod
    def create_random(cls, road, velocity=None, ego=False):
        l = random.randint(0,len(road.lanes)-1)
        xmin = np.min([v.position[0] for v in road.vehicles]) if len(road.vehicles) else 0
        offset = 30*np.exp(-5/30*len(road.lanes))
        v = Vehicle(road.lanes[l].position(xmin-offset, 0), 0, velocity, ego)
        return v

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
        s = pygame.Surface((screen.pix(self.LENGTH), screen.pix(self.LENGTH)), pygame.SRCALPHA)   # per-pixel alpha
        pygame.draw.rect(s, self.color, (0,screen.pix(self.LENGTH)/2-screen.pix(self.WIDTH)/2,screen.pix(self.LENGTH),screen.pix(self.WIDTH)), 0)
        pygame.draw.rect(s, screen.BLACK, (0,screen.pix(self.LENGTH)/2-screen.pix(self.WIDTH)/2,screen.pix(self.LENGTH),screen.pix(self.WIDTH)), 1)
        s = s.convert_alpha()
        h = self.heading if abs(self.heading) > 2*np.pi/180 else 0
        sr = pygame.transform.rotate(s, -h*180/np.pi)
        screen.blit(sr, (screen.pos2pix(self.position[0]-self.LENGTH/2, self.position[1]-self.LENGTH/2)))

    def display_trajectory(self, screen, states):
        for i in range(len(states)):
            s = states[i]
            s.color = (s.color[0], s.color[1], s.color[2], 50)
            s.display(screen)

    def __str__(self):
        return "[{}]: {}, {}, {}".format(self.id, self.position, self.heading, self.velocity)

    def __repr__(self):
        return self.__str__()

class ControlledVehicle(Vehicle):
    """
        A vehicle piloted by a low-level controller, allowing high-level actions
        such as lane changes.
    """
    TAU_A = 0.1
    TAU_DS = 0.2
    KP_A = 1/TAU_A
    KD_S = 1/TAU_DS
    KP_S = 0.1
    MAX_STEERING_ANGLE = np.pi/4

    def __init__(self, position, heading, velocity, ego, road, target_lane, target_velocity):
        super(ControlledVehicle, self).__init__(position, heading, velocity, ego)
        self.road = road
        self.target_lane = target_lane
        self.target_velocity = target_velocity

    @classmethod
    def create_from(cls, vehicle, road):
        return ControlledVehicle(vehicle.position, vehicle.heading, vehicle.velocity, vehicle.ego, road, road.get_lane_index(vehicle.position), vehicle.velocity)

    @classmethod
    def create_random(cls, road, velocity=None, ego=False):
        return cls.create_from(Vehicle.create_random(road, velocity, ego), road)

    def step(self, dt):
        action = {}
        # Lateral controller: lane keeping
        lane_coords = self.road.get_lane_coordinates(self.target_lane, self.position)
        heading_ref = -np.arctan(self.KP_S*lane_coords[1])*np.sign(self.velocity)+self.road.lanes[self.target_lane].heading_at(lane_coords[0]+self.velocity*self.TAU_DS)
        action['steering'] = self.KD_S*wrap_to_pi(heading_ref-self.heading)*self.LENGTH/self.velocity
        action['steering'] = min(max(action['steering'], -self.MAX_STEERING_ANGLE), self.MAX_STEERING_ANGLE)
        # Longitudinal controller: velocity control
        action['acceleration'] = self.KP_A*(self.target_velocity - self.velocity)

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
            next_lane = min(self.target_lane+1, len(self.road.lanes)-1)
            x,y = self.road.lanes[next_lane].local_coordinates(self.position)
            if abs(y) < 2*self.road.lanes[next_lane].width_at(x):
                self.target_lane  = next_lane
        elif action == "LANE_LEFT":
            next_lane = max(self.target_lane-1, 0)
            x,y = self.road.lanes[next_lane].local_coordinates(self.position)
            if abs(y) < 2*self.road.lanes[next_lane].width_at(x):
                self.target_lane  = next_lane


class MDPVehicle(ControlledVehicle):
    """
        A vehicle piloted by a low-level controller, allowing high-level actions
        such as lane changes.
    """

    SPEED_MIN = 25
    SPEED_COUNT = 1
    SPEED_MAX = 35

    def __init__(self, position, heading, velocity, ego, road, target_lane, target_velocity):
        super(MDPVehicle, self).__init__(position, heading, velocity, ego, road, target_lane, target_velocity)
        self.velocity_index = self.speed_to_index(target_velocity)
        self.target_velocity = self.index_to_speed(self.velocity_index)

    @classmethod
    def create_from(cls, vehicle, road):
        return MDPVehicle(vehicle.position, vehicle.heading, vehicle.velocity, vehicle.ego, road, road.get_lane_index(vehicle.position), vehicle.velocity)

    @classmethod
    def create_random(cls, road, velocity=None, ego=False):
        return cls.create_from(Vehicle.create_random(road, velocity, ego), road)

    def step(self, dt):
        self.target_velocity = self.index_to_speed(self.velocity_index)
        super(MDPVehicle, self).step(dt)


    @classmethod
    def index_to_speed(cls, index):
        if cls.SPEED_COUNT > 1:
            return cls.SPEED_MIN+index*(cls.SPEED_MAX-cls.SPEED_MIN)/(cls.SPEED_COUNT-1)
        else:
            return cls.SPEED_MIN

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
            self.target_lane = self.road.get_lane_index(self.position)+1
        elif action == "LANE_LEFT":
            self.target_lane = self.road.get_lane_index(self.position)-1

        self.velocity_index = min(max(self.velocity_index, 0), self.SPEED_COUNT-1)
        self.target_lane = min(max(self.target_lane, 0), len(self.road.lanes)-1)

    def predict_trajectory(self, actions, action_duration, log_duration, dt):
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.perform_action(action)
            for _ in range(int(action_duration/dt)):
                t+=1
                v.step(dt)
                if (t % int(log_duration/dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states


    def display(self, screen):
        super(ControlledVehicle, self).display(screen)

class OptionsVehicle(ControlledVehicle):
    """
        A vehicle piloted by a low-level controller, allowing high-level actions
        such as lane changes.
    """

    SPEED_MIN = 25
    SPEED_COUNT = 1
    SPEED_MAX = 35

    def __init__(self, position, heading, velocity, ego, road, target_lane, target_velocity):
        super(OptionsVehicle, self).__init__(position, heading, velocity, ego, road, target_lane, target_velocity)
        self.options = []

    @classmethod
    def create_from(cls, vehicle, road):
        return OptionsVehicle(vehicle.position, vehicle.heading, vehicle.velocity, vehicle.ego, road, road.get_lane_index(vehicle.position), vehicle.velocity)

    @classmethod
    def create_random(cls, road, velocity=None, ego=False):
        return cls.create_from(Vehicle.create_random(road, velocity, ego), road)

    def add_option_over_vehicle(self, vehicle, placement):
        self.options.append({'vehicle':vehicle, 'placement':placement})

    def step(self, dt):
        for o in self.options:
            v = o['vehicle']
            if o['placement'] == "TAKE_WAY":
                d = 1.0*v.velocity
            elif o['placement'] == "GIVE_WAY":
                d = -1.0*v.velocity
            else:
                d = 0
            l = self.road.get_lane(v.position)
            x = l.local_coordinates(self.position)[0]
            xv = l.local_coordinates(v.position)[0]
            Kp = 1./3
            self.target_velocity = v.velocity + Kp*(d + xv - x)

        super(OptionsVehicle, self).step(dt)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                self.perform_action("TAKE_WAY")
            if event.key == pygame.K_LEFT:
                self.perform_action("GIVE_WAY")

    def perform_action(self, action):
        for o in self.options:
            o['placement'] = action

    def predict_trajectory(self, actions, action_duration, log_duration, dt):
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.perform_action(action)
            for _ in range(int(action_duration/dt)):
                t+=1
                v.step(dt)
                if (t % int(log_duration/dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states


class IDMVehicle(Vehicle):
    """ Longitudinal controller that takes into account the front vehicle's distance and velocity.
        Two settings are possible: IDM and MAXIMUM_VELOCITY.
        The lateral controller is a lane keeping PD."""
    CONTROLLER_IDM = 0
    CONTROLLER_MAX_VELOCITY = 1

    # IDM parameters
    ACC_MAX = 3.0
    BRAKE_ACC = 5.0
    VELOCITY_WANTED = 20.0
    DISTANCE_WANTED = 6.0
    TIME_WANTED = 2.0
    DELTA = 4.0

    def __init__(self, position, heading, velocity, ego, road, target_lane, front_vehicle):
        super(IDMVehicle, self).__init__(position, heading, velocity, ego)
        self.road = road
        self.target_lane = target_lane
        self.front_vehicle = front_vehicle
        self.color = Vehicle.BLUE
        self.controller = self.CONTROLLER_IDM

    @classmethod
    def create_from(cls, vehicle, road):
        return IDMVehicle(vehicle.position, vehicle.heading, vehicle.velocity, vehicle.ego, road, road.get_lane_index(vehicle.position), None)

    @classmethod
    def create_random(cls, road, velocity=None, ego=False):
        return cls.create_from(Vehicle.create_random(road, velocity, ego), road)

    def step(self, dt):
        action = {}
        # Lateral controller: lane keeping
        lane_coords = self.road.get_lane_coordinates(self.target_lane, self.position)
        heading_ref = -np.arctan(ControlledVehicle.KP_S*lane_coords[1])+self.road.lanes[self.target_lane].heading_at(lane_coords[0]+self.velocity*ControlledVehicle.TAU_DS)
        action['steering'] = ControlledVehicle.KD_S*wrap_to_pi(heading_ref-self.heading)*self.LENGTH/max(self.velocity,1)
        action['steering'] = min(max(action['steering'], -ControlledVehicle.MAX_STEERING_ANGLE), ControlledVehicle.MAX_STEERING_ANGLE)

        # Intelligent Driver Model
        if self.controller == self.CONTROLLER_IDM:
            action['acceleration'] = self.ACC_MAX*(1-np.power(self.velocity/self.VELOCITY_WANTED, self.DELTA))
            self.front_vehicle = self.road.front_vehicle(self)
            if self.front_vehicle:
                l = self.road.get_lane(self.position)
                d = max(l.local_coordinates(self.front_vehicle.position)[0] - l.local_coordinates(self.position)[0], 1)
                d_star = self.DISTANCE_WANTED + self.velocity*self.TIME_WANTED + \
                    self.velocity*(self.velocity-self.front_vehicle.velocity)/(2*np.sqrt(self.ACC_MAX*self.BRAKE_ACC))
                action['acceleration'] -= self.ACC_MAX*np.power(d_star/d,2)

        # Max velocity
        if self.controller == self.CONTROLLER_MAX_VELOCITY:
            action['acceleration'] = 2*ControlledVehicle.KP_A*(self.maximum_velocity() - self.velocity)

        action['acceleration'] = min(max(action['acceleration'], -self.BRAKE_ACC), self.ACC_MAX)
        super(IDMVehicle, self).step(dt, action)

    def maximum_velocity(self):
        if not self.front_vehicle:
            return self.VELOCITY_WANTED
        a0 = self.BRAKE_ACC
        a1 = self.BRAKE_ACC
        tau = 1.0
        l = self.road.get_lane(self.position)
        d = max(l.local_coordinates(self.front_vehicle.position)[0] - l.local_coordinates(self.position)[0] - self.DISTANCE_WANTED, 0)
        v1_0 = self.front_vehicle.velocity
        delta = 4*(a0*a1*tau)**2+8*a0*(a1**2)*d+4*a0*a1*v1_0**2
        return -a0*tau+np.sqrt(delta)/(2*a1)

def test():
    from simulation import Simulation
    from road import Road
    road = Road.create_random_road(lanes_count=2, lane_width=4.0, vehicles_count=10, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=ControlledVehicle)
    # other = sim.road.vehicles[0]
    # sim.vehicle.add_option_over_vehicle(other, 'TAKE_WAY')

    while not sim.done:
        sim.process()
    sim.quit()

if __name__ == '__main__':
    test()