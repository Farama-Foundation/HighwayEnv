from __future__ import division, print_function
import numpy as np
import pygame
import random
import copy
from highway import utils


class Vehicle(object):
    """
        A moving vehicle and its dynamics.
    """
    LENGTH = 5.0
    WIDTH = 2.0
    STEERING_TAU = 0.2

    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    BLUE = (100, 200, 255)
    BLACK = (60, 60, 60)

    id_max = 0

    def __init__(self, road, position, heading=0, velocity=None, ego=False):
        self.road = road
        self.position = np.array(position)
        self.heading = heading
        self.steering_angle = 0
        if velocity is None:
            velocity = 20 - random.randint(0, 3)
        self.velocity = velocity
        self.ego = ego
        self.color = self.GREEN if self.ego else self.YELLOW
        self.action = {'steering': 0, 'acceleration': 0}

        self.lane = None
        self.id = Vehicle.id_max
        Vehicle.id_max += 1

    @classmethod
    def create_random(cls, road, velocity=None, ego=False):
        lane = random.randint(0, len(road.lanes) - 1)
        x_min = np.min([v.position[0] for v in road.vehicles]) if len(road.vehicles) else 0
        offset = 30 * np.exp(-5 / 30 * len(road.lanes))
        v = Vehicle(road, road.lanes[lane].position(x_min - offset, 0), 0, velocity, ego)
        return v

    def step(self, dt, action=None):
        if not action:
            action = self.action
        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * dt
        self.heading += self.velocity * self.steering_angle / self.LENGTH * dt
        self.steering_angle += 1 / self.STEERING_TAU * (np.tan(action['steering']) - self.steering_angle) * dt
        self.velocity += action['acceleration'] * dt

    def lane_distance_to_vehicle(self, vehicle):
        lane = self.road.get_lane(self.position)
        return lane.local_coordinates(vehicle.position)[0] - lane.local_coordinates(self.position)[0]

    def handle_event(self, event):
        if not self.action:
            self.action = {}
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                self.action['acceleration'] = 4
            if event.key == pygame.K_LEFT:
                self.action['acceleration'] = -6
            if event.key == pygame.K_DOWN:
                self.action['steering'] = 20 * np.pi / 180
            if event.key == pygame.K_UP:
                self.action['steering'] = -20 * np.pi / 180
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
        s = pygame.Surface((screen.pix(self.LENGTH), screen.pix(self.LENGTH)), pygame.SRCALPHA)  # per-pixel alpha
        pygame.draw.rect(s, self.color, (0,
                                         screen.pix(self.LENGTH) / 2 - screen.pix(self.WIDTH) / 2,
                                         screen.pix(self.LENGTH), screen.pix(self.WIDTH)),
                         0)
        pygame.draw.rect(s, screen.BLACK, (0,
                                           screen.pix(self.LENGTH) / 2 - screen.pix(self.WIDTH) / 2,
                                           screen.pix(self.LENGTH), screen.pix(self.WIDTH)),
                         1)
        s = pygame.Surface.convert_alpha(s)
        h = self.heading if abs(self.heading) > 2 * np.pi / 180 else 0
        sr = pygame.transform.rotate(s, -h * 180 / np.pi)
        screen.blit(sr, (screen.pos2pix(self.position[0] - self.LENGTH / 2, self.position[1] - self.LENGTH / 2)))

    @staticmethod
    def display_trajectory(screen, states):
        for i in range(len(states)):
            s = states[i]
            s.color = (s.color[0], s.color[1], s.color[2], 50)
            s.display(screen)

    def __str__(self):
        return "#{}:{}".format(self.id, self.position)

    def __repr__(self):
        return self.__str__()


class Obstacle(Vehicle):
    def __init__(self, road, position):
        super(Obstacle, self).__init__(road, position, velocity=0)
        self.color = Vehicle.BLACK
        self.LENGTH = self.WIDTH
        self.target_velocity = 0


class ControlledVehicle(Vehicle):
    """
        A vehicle piloted by a low-level controller, allowing high-level actions
        such as lane changes.
    """
    TAU_A = 0.1
    TAU_DS = 0.3
    KP_A = 1 / TAU_A
    KD_S = 1 / TAU_DS
    KP_S = 0.05
    MAX_STEERING_ANGLE = np.pi / 4

    def __init__(self, road, position, heading=0, velocity=None, ego=False, target_lane=None, target_velocity=None):
        super(ControlledVehicle, self).__init__(road, position, heading, velocity, ego)
        self.target_lane = target_lane or road.get_lane_index(self.position)
        self.target_velocity = target_velocity or self.velocity

    @classmethod
    def create_from(cls, vehicle):
        return ControlledVehicle(vehicle.road, vehicle.position, vehicle.heading, vehicle.velocity, vehicle.ego, None,
                                 None)

    @classmethod
    def create_random(cls, road, velocity=None, ego=False):
        return cls.create_from(Vehicle.create_random(road, velocity, ego))

    def step(self, dt, action=None):
        action = {'steering': self.steering_control(self.target_lane),
                  'acceleration': self.velocity_control(self.target_velocity)}
        super(ControlledVehicle, self).step(dt, action)

    def steering_control(self, target_lane):
        lane_coords = self.road.get_lane_coordinates(target_lane, self.position)
        lane_next_coords = lane_coords[0] + self.velocity * (self.TAU_DS + Vehicle.STEERING_TAU)
        lane_future_heading = self.road.lanes[target_lane].heading_at(lane_next_coords)
        heading_ref = -np.arctan(self.KP_S * lane_coords[1]) * np.sign(self.velocity) + lane_future_heading
        steering = self.KD_S * utils.wrap_to_pi(heading_ref - self.heading) * self.LENGTH / utils.not_zero(
            self.velocity)
        steering = utils.constrain(steering, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering

    def velocity_control(self, target_velocity):
        return self.KP_A * (target_velocity - self.velocity)

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
            next_lane = min(self.target_lane + 1, len(self.road.lanes) - 1)
            x, y = self.road.lanes[next_lane].local_coordinates(self.position)
            if abs(y) < 2 * self.road.lanes[next_lane].width_at(x):
                self.target_lane = next_lane
        elif action == "LANE_LEFT":
            next_lane = max(self.target_lane - 1, 0)
            x, y = self.road.lanes[next_lane].local_coordinates(self.position)
            if abs(y) < 2 * self.road.lanes[next_lane].width_at(x):
                self.target_lane = next_lane


class MDPVehicle(ControlledVehicle):
    """
        A vehicle piloted by a low-level controller, allowing high-level actions
        such as lane changes.
    """

    SPEED_MIN = 25
    SPEED_COUNT = 1
    SPEED_MAX = 35

    def __init__(self, road, position, heading=0, velocity=None, ego=False, target_lane=None, target_velocity=None):
        super(MDPVehicle, self).__init__(road, position, heading, velocity, ego, target_lane, target_velocity)
        self.velocity_index = self.speed_to_index(self.target_velocity)
        self.target_velocity = self.index_to_speed(self.velocity_index)

    @classmethod
    def create_from(cls, vehicle):
        return MDPVehicle(vehicle.road, vehicle.position, vehicle.heading, vehicle.velocity, vehicle.ego, None, None)

    @classmethod
    def create_random(cls, road, velocity=None, ego=False):
        return cls.create_from(Vehicle.create_random(road, velocity, ego))

    def step(self, dt, action=None):
        self.target_velocity = self.index_to_speed(self.velocity_index)
        super(MDPVehicle, self).step(dt)

    @classmethod
    def index_to_speed(cls, index):
        if cls.SPEED_COUNT > 1:
            return cls.SPEED_MIN + index * (cls.SPEED_MAX - cls.SPEED_MIN) / (cls.SPEED_COUNT - 1)
        else:
            return cls.SPEED_MIN

    @classmethod
    def speed_to_index(cls, speed):
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.round(x * (cls.SPEED_COUNT - 1)))

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
            self.target_lane = self.road.get_lane_index(self.position) + 1
        elif action == "LANE_LEFT":
            self.target_lane = self.road.get_lane_index(self.position) - 1

        self.velocity_index = min(max(self.velocity_index, 0), self.SPEED_COUNT - 1)
        self.target_lane = min(max(self.target_lane, 0), len(self.road.lanes) - 1)

    def predict_trajectory(self, actions, action_duration, log_duration, dt):
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.perform_action(action)
            for _ in range(int(action_duration / dt)):
                t += 1
                v.step(dt)
                if (t % int(log_duration / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states

    def display(self, screen):
        super(ControlledVehicle, self).display(screen)


class IDMVehicle(ControlledVehicle):
    """ Longitudinal controller that takes into account the front vehicle's distance and velocity.
        Two settings are possible: IDM and MAXIMUM_VELOCITY.
        The lateral controller is a lane keeping PD."""
    CONTROLLER_IDM = 0
    CONTROLLER_MAX_VELOCITY = 1

    # Longitudinal control parameters
    ACC_MAX = 3.0
    BRAKE_ACC = 5.0
    VELOCITY_WANTED = 20.0
    DISTANCE_WANTED = 5.0
    TIME_WANTED = 1.0
    DELTA = 4.0

    # Lane change parameters
    LANE_CHANGE_MIN_ACC_GAIN = 0.2
    LANE_CHANGE_MAX_ACC_LOSS = 3.
    LANE_CHANGE_AVG_DELAY = 0.5

    def __init__(self, road, position, heading=0, velocity=None, ego=False, target_lane=None):
        super(IDMVehicle, self).__init__(road, position, heading, velocity, ego)
        self.target_lane = target_lane or road.get_lane_index(self.position)
        self.color = Vehicle.BLUE
        self.target_velocity = self.VELOCITY_WANTED + random.randint(-5, 5)
        self.controller = self.CONTROLLER_IDM

    @classmethod
    def create_from(cls, vehicle):
        return IDMVehicle(vehicle.road, vehicle.position, vehicle.heading, vehicle.velocity, vehicle.ego, None)

    @classmethod
    def create_random(cls, road, velocity=None, ego=False):
        return cls.create_from(Vehicle.create_random(road, velocity, ego))

    def step(self, dt, action=None):
        action = {}

        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        # Lateral controller: lane keeping
        self.change_lane_policy(dt)
        action['steering'] = self.steering_control(self.target_lane)

        # Intelligent Driver Model
        if self.controller == self.CONTROLLER_IDM:
            action['acceleration'] = IDMVehicle.idm(ego_vehicle=self, front_vehicle=front_vehicle)

        # Max velocity
        if self.controller == self.CONTROLLER_MAX_VELOCITY:
            self.target_velocity = min(self.maximum_velocity(front_vehicle), self.target_velocity)
            action['acceleration'] = self.velocity_control(self.target_velocity)

        action['acceleration'] = utils.constrain(action['acceleration'], -self.BRAKE_ACC, self.ACC_MAX)
        super(ControlledVehicle, self).step(dt, action)

    @classmethod
    def idm(cls, ego_vehicle, front_vehicle=None):
        """
            Compute an acceleration command with the Intelligent Driver Model.

            The acceleration is chosen so as to:
            - reach a target velocity
            - maintain a minimum safety distance (and safety time) w.r.t the front vehicle
        """
        if not ego_vehicle:
            raise Exception("A vehicle should be provided to compute its longitudinal acceleration")
        acceleration = cls.ACC_MAX * (
                1 - np.power(ego_vehicle.velocity / utils.not_zero(ego_vehicle.target_velocity), cls.DELTA))
        if front_vehicle:
            d0 = cls.DISTANCE_WANTED
            tau = cls.TIME_WANTED
            ab = cls.ACC_MAX * cls.BRAKE_ACC
            d = ego_vehicle.lane_distance_to_vehicle(front_vehicle) - ego_vehicle.LENGTH / 2 - front_vehicle.LENGTH / 2
            dv = ego_vehicle.velocity - front_vehicle.velocity
            d_star = d0 + ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab))
            acceleration -= cls.ACC_MAX * np.power(d_star / utils.not_zero(d), 2)
        return acceleration

    def maximum_velocity(self, front_vehicle=None):
        """
            Compute the maximum allowed velocity to avoid Inevitable Collision States.

            Assume the front vehicle is going to brake at full deceleration and that
            it will be noticed after a given delay, and compute the maximum velocity
            which allows the ego-vehicle to brake enough to avoid the collision.
        """
        if not front_vehicle:
            return self.target_velocity
        d0 = self.DISTANCE_WANTED
        a0 = self.BRAKE_ACC
        a1 = self.BRAKE_ACC
        tau = self.TIME_WANTED
        d = max(self.lane_distance_to_vehicle(front_vehicle) - self.LENGTH / 2 - front_vehicle.LENGTH / 2 - d0, 0)
        v1_0 = front_vehicle.velocity
        delta = 4 * (a0 * a1 * tau) ** 2 + 8 * a0 * (a1 ** 2) * d + 4 * a0 * a1 * v1_0 ** 2
        v_max = -a0 * tau + np.sqrt(delta) / (2 * a1)
        return v_max

    def change_lane_policy(self, dt):
        """
            Make a lane change decision
            - only once in a while
            - only if the lane change is relevant
        """
        if not utils.do_on_average_every(self.LANE_CHANGE_AVG_DELAY, dt):
            return

        # Check if we should change to an adjacent lane
        lanes = utils.constrain(self.target_lane + np.array([-1, 1]), 0, len(self.road.lanes) - 1)
        for lane in lanes:
            if lane != self.target_lane and self.should_change_lane(lane):
                self.target_lane = lane

    def should_change_lane(self, lane_index):
        """
            Decide whether a change to a given lane is relevant:
            - The lane should be close enough from the vehicle.
            - After changing I should be able to accelerate more
            - I should not impose too big an acceleration on the target lane rear vehicle.
        """
        # Is the target lane close enough?
        x, y = self.road.lanes[lane_index].local_coordinates(self.position)
        lane_close = abs(y) < 2 * self.road.lanes[lane_index].width_at(x)
        if not lane_close:
            return False

        # Is there an advantage for me to change lane?
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        lane_front_vehicle, lane_rear_vehicle = self.road.neighbour_vehicles(self, self.road.lanes[lane_index])
        current_acceleration = IDMVehicle.idm(ego_vehicle=self, front_vehicle=front_vehicle)
        predicted_acceleration = IDMVehicle.idm(ego_vehicle=self, front_vehicle=lane_front_vehicle)
        self_advantage = predicted_acceleration - current_acceleration > self.LANE_CHANGE_MIN_ACC_GAIN
        if not self_advantage:
            return False

        # Is there a disadvantage for the rear vehicle in target lane?
        if lane_rear_vehicle:
            current_acceleration = IDMVehicle.idm(ego_vehicle=lane_rear_vehicle, front_vehicle=lane_front_vehicle)
            predicted_acceleration = IDMVehicle.idm(ego_vehicle=lane_rear_vehicle, front_vehicle=self)
            lane_rear_disadvantage = predicted_acceleration - current_acceleration < -self.LANE_CHANGE_MAX_ACC_LOSS
            if lane_rear_disadvantage:
                return False

        # All clear, let's go!
        return True


def test():
    from highway.simulation import Simulation
    from highway.road import Road
    road = Road.create_random_road(lanes_count=2, lane_width=4.0, vehicles_count=3, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=ControlledVehicle)

    while not sim.done:
        sim.process()
    sim.quit()


if __name__ == '__main__':
    test()
