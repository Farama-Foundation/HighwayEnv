from __future__ import division, print_function
import numpy as np
import pygame
import copy
from highway import utils


class Vehicle(object):
    """
        A moving vehicle and its dynamics.
    """
    LENGTH = 5.0  # [m]
    WIDTH = 2.0  # [m]
    STEERING_TAU = 0.2  # [s]
    ENABLE_CRASHES = True

    DEFAULT_VELOCITIES = [20, 25]  # [m/s]

    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN

    id_max = 0

    def __init__(self, road, position, heading=0, velocity=0):
        self.road = road
        self.position = np.array(position)
        self.heading = heading
        self.steering_angle = 0
        self.velocity = velocity
        self.lane_index = self.road.get_lane_index(self.position)
        self.lane = self.road.lanes[self.lane_index]
        self.color = self.DEFAULT_COLOR
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False

        self.id = Vehicle.id_max
        Vehicle.id_max += 1

    @classmethod
    def create_random(cls, road, velocity=None):
        lane = np.random.randint(0, len(road.lanes) - 1)
        x_min = np.min([v.position[0] for v in road.vehicles]) if len(road.vehicles) else 0
        offset = 30 * np.exp(-5 / 30 * len(road.lanes))
        velocity = velocity or np.random.randint(Vehicle.DEFAULT_VELOCITIES[0], Vehicle.DEFAULT_VELOCITIES[1])
        v = Vehicle(road, road.lanes[lane].position(x_min - offset, 0), 0, velocity)
        return v

    def act(self, action=None):
        if action:
            self.action = action

    def step(self, dt):
        if self.crashed:
            self.action['steering'] = np.pi / 4 * (-1 + 2 * np.random.beta(0.5, 0.5))
            self.action['acceleration'] = (-1.0 + 0.2 * np.random.beta(0.5, 0.5)) * self.velocity

        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * dt
        self.heading += self.velocity * self.steering_angle / self.LENGTH * dt
        self.steering_angle += 1 / self.STEERING_TAU * (np.tan(self.action['steering']) - self.steering_angle) * dt
        self.velocity += self.action['acceleration'] * dt

        self.lane_index = self.road.get_lane_index(self.position)
        self.lane = self.road.lanes[self.lane_index]

    def lane_distance_to_vehicle(self, vehicle):
        if not vehicle:
            return np.nan
        return self.lane.local_coordinates(vehicle.position)[0] - self.lane.local_coordinates(self.position)[0]

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

    def check_collision(self, other):
        """Check for collision with another vehicle.

        :param other: the other vehicle
        """
        if not self.ENABLE_CRASHES:
            return
        if other is self:
            return
        if np.linalg.norm(self.position - other.position) < self.WIDTH/2 + other.WIDTH/2:
            self.crashed = other.crashed = True
            self.color = other.color = Vehicle.RED

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
    TAU_A = 0.1  # [s]
    TAU_DS = 0.3  # [s]
    KP_A = 1 / TAU_A
    KD_S = 1 / TAU_DS
    KP_S = 0.05  # [1/m]
    MAX_STEERING_ANGLE = np.pi / 4  # [rad]
    STEERING_VEL_GAIN = 60  # [m/s]

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None):
        super(ControlledVehicle, self).__init__(road, position, heading, velocity)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_velocity = target_velocity or self.velocity

    @classmethod
    def create_from(cls, vehicle):
        return ControlledVehicle(vehicle.road, vehicle.position, vehicle.heading, vehicle.velocity, None, None)

    @classmethod
    def create_random(cls, road, velocity=None):
        return cls.create_from(Vehicle.create_random(road, velocity))

    def act(self, action=None):
        if action == "FASTER":
            self.target_velocity += 5
        elif action == "SLOWER":
            self.target_velocity -= 5
        elif action == "LANE_RIGHT":
            target_lane_index = self.lane_index + 1
            if target_lane_index < len(self.road.lanes) and \
                    self.road.lanes[target_lane_index].is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            target_lane_index = self.lane_index - 1
            if target_lane_index >= 0 and self.road.lanes[target_lane_index].is_reachable_from(self.position):
                self.target_lane_index = target_lane_index

        action = {'steering': self.steering_control(self.target_lane_index),
                  'acceleration': self.velocity_control(self.target_velocity)}
        super(ControlledVehicle, self).act(action)

    def steering_control(self, target_lane_index):
        lane_coords = self.road.lanes[target_lane_index].local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * (self.TAU_DS + Vehicle.STEERING_TAU)
        lane_future_heading = self.road.lanes[target_lane_index].heading_at(lane_next_coords)
        heading_ref = -np.arctan(self.KP_S * lane_coords[1] * np.exp(-(self.velocity/self.STEERING_VEL_GAIN)**2)) \
            * np.sign(self.velocity) + lane_future_heading
        steering = self.KD_S * utils.wrap_to_pi(heading_ref - self.heading) * self.LENGTH / utils.not_zero(
            self.velocity)
        steering = utils.constrain(steering, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering

    def velocity_control(self, target_velocity):
        return self.KP_A * (target_velocity - self.velocity)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                self.act("FASTER")
            if event.key == pygame.K_LEFT:
                self.act("SLOWER")
            if event.key == pygame.K_DOWN:
                self.act("LANE_RIGHT")
            if event.key == pygame.K_UP:
                self.act("LANE_LEFT")


class MDPVehicle(ControlledVehicle):
    """
        A vehicle piloted by a low-level controller, allowing high-level actions
        such as lane changes.
    """

    SPEED_COUNT = 1  # []
    SPEED_MIN = 25  # [m/s]
    SPEED_MAX = 35  # [m/s]

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=None,
                 target_lane_index=None,
                 target_velocity=None):
        super(MDPVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity)
        self.velocity_index = self.speed_to_index(self.target_velocity)
        self.target_velocity = self.index_to_speed(self.velocity_index)

    @classmethod
    def create_from(cls, vehicle):
        return MDPVehicle(vehicle.road, vehicle.position, vehicle.heading, vehicle.velocity, None, None)

    @classmethod
    def create_random(cls, road, velocity=None):
        return cls.create_from(Vehicle.create_random(road, velocity))

    def act(self, action=None):
        if action == "FASTER":
            self.velocity_index = self.speed_to_index(self.velocity) + 1
        elif action == "SLOWER":
            self.velocity_index = self.speed_to_index(self.velocity) - 1
        elif action == "LANE_RIGHT":
            target_lane_index = self.lane_index + 1
            if target_lane_index < len(self.road.lanes) and \
                    self.road.lanes[target_lane_index].is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            target_lane_index = self.lane_index - 1
            if target_lane_index >= 0 and self.road.lanes[target_lane_index].is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        self.velocity_index = utils.constrain(self.velocity_index, 0, self.SPEED_COUNT - 1)
        self.target_velocity = self.index_to_speed(self.velocity_index)

        super(MDPVehicle, self).act()

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

    def predict_trajectory(self, actions, action_duration, log_duration, dt):
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
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
    ACC_MAX = 3.0  # [m/s2]
    BRAKE_ACC = 5.0  # [m/s2]
    VELOCITY_WANTED = 20.0  # [m/s]
    DISTANCE_WANTED = 5.0  # [m]
    TIME_WANTED = 1.0  # [s]
    DELTA = 4.0  # []

    # Lane change parameters
    POLITENESS = 0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    IDM_COLOR = Vehicle.BLUE

    def __init__(self, road, position, heading=0, velocity=0, target_lane_index=None):
        super(IDMVehicle, self).__init__(road, position, heading, velocity, target_lane_index)
        self.color = self.IDM_COLOR
        self.target_velocity = self.VELOCITY_WANTED
        self.controller = self.CONTROLLER_IDM
        self.timer = np.random.random()*self.LANE_CHANGE_DELAY

    @classmethod
    def create_from(cls, vehicle):
        return IDMVehicle(vehicle.road, vehicle.position, vehicle.heading, vehicle.velocity, None)

    @classmethod
    def create_random(cls, road, velocity=None):
        return cls.create_from(Vehicle.create_random(road, velocity))

    def act(self, action=None):
        action = {}

        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        # Lateral controller: lane keeping
        self.change_lane_policy()
        action['steering'] = self.steering_control(self.target_lane_index)

        # Intelligent Driver Model
        if self.controller == self.CONTROLLER_IDM:
            action['acceleration'] = IDMVehicle.idm(ego_vehicle=self, front_vehicle=front_vehicle)

        # Max velocity
        if self.controller == self.CONTROLLER_MAX_VELOCITY:
            self.target_velocity = min(self.maximum_velocity(front_vehicle), self.target_velocity)
            action['acceleration'] = self.velocity_control(self.target_velocity)

        action['acceleration'] = utils.constrain(action['acceleration'], -self.BRAKE_ACC, self.ACC_MAX)
        super(ControlledVehicle, self).act(action)

    def step(self, dt):
        self.timer += dt
        super(IDMVehicle, self).step(dt)

    @classmethod
    def idm(cls, ego_vehicle, front_vehicle=None):
        """
            Compute an acceleration command with the Intelligent Driver Model.

            The acceleration is chosen so as to:
            - reach a target velocity
            - maintain a minimum safety distance (and safety time) w.r.t the front vehicle
        """
        if not ego_vehicle:
            return 0
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

    def change_lane_policy(self):
        """
            Make a lane change decision
            - only once in a while
            - only if the lane change is relevant
        """
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # Check if we should change to an adjacent lane
        lane_indexes = utils.constrain(self.road.get_lane_index(self.position) + np.array([-1, 1]),
                                       0, len(self.road.lanes) - 1)
        for lane_index in lane_indexes:
            if lane_index != self.target_lane_index and self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index):
        """
            MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change
            Returns whether the vehicle should change lane, if:
            - The target lane is close enough.
            - After changing it (and/or following vehicles) can accelerate more
            - It doesn't impose an unsafe braking on its new following vehicle.
        """
        # Is the target lane close enough?
        if not self.road.lanes[lane_index].is_reachable_from(self.position):
            return False

        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, self.road.lanes[lane_index])
        new_following_a = IDMVehicle.idm(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = IDMVehicle.idm(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Is there an advantage for me and/or my followers to change lane?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_a = IDMVehicle.idm(ego_vehicle=self, front_vehicle=old_preceding)
        self_pred_a = IDMVehicle.idm(ego_vehicle=self, front_vehicle=new_preceding)
        old_following_a = IDMVehicle.idm(ego_vehicle=old_following, front_vehicle=self)
        old_following_pred_a = IDMVehicle.idm(ego_vehicle=old_following, front_vehicle=old_preceding)
        jerk = self_pred_a - self_a + self.POLITENESS*(new_following_pred_a - new_following_a
                                                       + old_following_pred_a - old_following_a)
        if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
            return False

        # All clear, let's go!
        return True


def test():
    from highway.simulation import Simulation
    from highway.road import Road
    road = Road.create_random_road(lanes_count=2, lane_width=4.0, vehicles_count=30, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=ControlledVehicle)

    while not sim.done:
        sim.process()
    sim.quit()


if __name__ == '__main__':
    test()
