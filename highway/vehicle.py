from __future__ import division, print_function
import numpy as np
import pandas as pd
import pygame
import copy
from highway import utils
from highway.logger import Loggable


class Vehicle(Loggable):
    """
        A moving vehicle on a road, and its dynamics.

        The vehicle is represented by a dynamical system: a modified bicycle model.
        It's state is propagated depending on its steering and acceleration actions.
    """
    COLLISIONS_ENABLED = True

    LENGTH = 5.0  # [m]
    WIDTH = 2.0  # [m]
    STEERING_TAU = 0.2  # [s]
    DEFAULT_VELOCITIES = [20, 25]  # [m/s]

    # Display
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN

    def __init__(self, road, position, heading=0, velocity=0):
        self.road = road
        self.position = np.array(position)
        self.heading = heading
        self.steering_angle = 0
        self.velocity = velocity
        self.lane_index = self.road.get_lane_index(self.position) if self.road else np.nan
        self.lane = self.road.lanes[self.lane_index] if self.road else np.nan
        self.color = self.DEFAULT_COLOR
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.log = []

    @classmethod
    def create_random(cls, road, velocity=None):
        """
            Create a random vehicle on the road.

            The lane and /or velocity are chosen randomly, while longitudinal position is chosen behind the last
            vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :return: A vehicle with random position and/or velocity
        """
        lane = np.random.randint(0, len(road.lanes))
        x_min = np.min([v.position[0] for v in road.vehicles]) if len(road.vehicles) else 0
        offset = 30 * np.exp(-5 / 30 * len(road.lanes))
        velocity = velocity or np.random.randint(Vehicle.DEFAULT_VELOCITIES[0], Vehicle.DEFAULT_VELOCITIES[1])
        v = cls(road, road.lanes[lane].position(x_min - offset, 0), 0, velocity)
        return v

    def act(self, action=None):
        """
            Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt):
        """
            Propagate the vehicle state given its actions.

            Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
            If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
            The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        if self.crashed:
            self.action['steering'] = np.pi / 4 * (-1 + 2 * np.random.beta(0.5, 0.5)) * 0
            self.action['acceleration'] = (-1.0 + 0.2 * np.random.beta(0.5, 0.5) * 0) * self.velocity

        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * dt
        self.heading += self.velocity * self.steering_angle / self.LENGTH * dt
        self.steering_angle += 1 / self.STEERING_TAU * (np.tan(self.action['steering']) - self.steering_angle) * dt
        self.velocity += self.action['acceleration'] * dt

        self.lane_index = self.road.get_lane_index(self.position)
        self.lane = self.road.lanes[self.lane_index]

    def lane_distance_to(self, vehicle):
        """
            Compute the signed distance to another vehicle along current lane.

        :param vehicle: the other vehicle
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        return self.lane.local_coordinates(vehicle.position)[0] - self.lane.local_coordinates(self.position)[0]

    def handle_event(self, event):
        """
            Handle a pygame event.

            The vehicle actions can be manually controlled.
        :param event: the pygame event
        """
        action = self.action
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                action['acceleration'] = 4
            if event.key == pygame.K_LEFT:
                action['acceleration'] = -6
            if event.key == pygame.K_DOWN:
                action['steering'] = 20 * np.pi / 180
            if event.key == pygame.K_UP:
                action['steering'] = -20 * np.pi / 180
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT:
                action['acceleration'] = 0
            if event.key == pygame.K_LEFT:
                action['acceleration'] = 0
            if event.key == pygame.K_DOWN:
                action['steering'] = 0
            if event.key == pygame.K_UP:
                action['steering'] = 0
        if action != self.action:
            self.act(action)

    def check_collision(self, other):
        """
            Check for collision with another vehicle.

        :param other: the other vehicle
        """
        if not self.COLLISIONS_ENABLED or self.crashed or other is self:
            return

        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return

        # Accurate elliptic check
        u = other.position - self.position
        c, s = np.cos(self.heading), np.sin(self.heading)
        r = np.matrix([[c, -s], [s, c]])
        ru = r.dot(u)
        if np.sum(np.square(ru/np.array([self.LENGTH, self.WIDTH]))) < 1:
            self.velocity = other.velocity = min(self.velocity, other.velocity)
            self.crashed = other.crashed = True
            self.color = other.color = Vehicle.RED

    def display(self, surface):
        """
            Display the vehicle on a pygame surface.

            The vehicle is represented as a colored rotated rectangle.

        :param surface: the surface to draw the vehicle on
        """
        s = pygame.Surface((surface.pix(self.LENGTH), surface.pix(self.LENGTH)), pygame.SRCALPHA)  # per-pixel alpha
        pygame.draw.rect(s, self.color, (0,
                                         surface.pix(self.LENGTH) / 2 - surface.pix(self.WIDTH) / 2,
                                         surface.pix(self.LENGTH), surface.pix(self.WIDTH)),
                         0)
        pygame.draw.rect(s, surface.BLACK, (0,
                                            surface.pix(self.LENGTH) / 2 - surface.pix(self.WIDTH) / 2,
                                            surface.pix(self.LENGTH), surface.pix(self.WIDTH)),
                         1)
        s = pygame.Surface.convert_alpha(s)
        h = self.heading if abs(self.heading) > 2 * np.pi / 180 else 0
        sr = pygame.transform.rotate(s, -h * 180 / np.pi)
        surface.blit(sr, (surface.pos2pix(self.position[0] - self.LENGTH / 2, self.position[1] - self.LENGTH / 2)))

    def dump(self):
        """
            Update the internal log of the vehicle, containing:
                - its kinematics;
                - some metrics relative to its neighbour vehicles.
        """
        data = {
            'v': self.velocity,
            'acceleration': self.action['acceleration'],
            'steering': self.action['steering']}

        if self.road:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
            if front_vehicle:
                data.update({
                    'front_v': front_vehicle.velocity,
                    'front_distance': self.lane_distance_to(front_vehicle)
                })
            if rear_vehicle:
                data.update({
                    'rear_v': rear_vehicle.velocity,
                    'rear_distance': rear_vehicle.lane_distance_to(self)
                })

        self.log.append(data)

    def get_log(self):
        """
            Cast the internal log as a DataFrame.

        :return: the DataFrame of the Vehicle's log.
        """
        return pd.DataFrame(self.log)

    @staticmethod
    def display_trajectory(surface, states):
        """
            Display the whole trajectory of a vehicle on a pygame surface.

        :param surface: the surface to draw the vehicle future states on
        :param states: the list of vehicle states within the trajectory
        """
        for i in range(len(states)):
            s = states[i]
            s.color = (s.color[0], s.color[1], s.color[2], 50)  # Color is made transparent
            s.display(surface)

    def __str__(self):
        return "#{}: {}".format(id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()


class Obstacle(Vehicle):
    """
        A motionless obstacle at a given position.
    """

    def __init__(self, road, position):
        super(Obstacle, self).__init__(road, position, velocity=0)
        self.target_velocity = 0
        self.color = Vehicle.BLACK
        self.LENGTH = self.WIDTH


class ControlledVehicle(Vehicle):
    """
        A vehicle piloted by two low-level controller, allowing high-level actions
        such as cruise control and lane changes.

        - The longitudinal controller is a velocity controller;
        - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    TAU_A = 0.3  # [s]
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

    def act(self, action=None):
        """
            Perform a high-level action to change the desired lane or velocity.

            - If a high-level action is provided, update the target velocity and lane;
            - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        if action == "FASTER":
            self.target_velocity += 5
        elif action == "SLOWER":
            self.target_velocity -= 5
        elif action == "LANE_RIGHT":
            target_lane_index = np.clip(self.lane_index + 1, 0, len(self.road.lanes) - 1)
            if self.road.lanes[target_lane_index].is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            target_lane_index = np.clip(self.lane_index - 1, 0, len(self.road.lanes) - 1)
            if self.road.lanes[target_lane_index].is_reachable_from(self.position):
                self.target_lane_index = target_lane_index

        action = {'steering': self.steering_control(self.target_lane_index),
                  'acceleration': self.velocity_control(self.target_velocity)}
        super(ControlledVehicle, self).act(action)

    def steering_control(self, target_lane_index):
        """
            Steer the vehicle to follow the center of an given lane.

            The steering command is computed by a proportional heading controller, whose heading reference is set to the
            target lane heading added with a proportional lateral position controller weighted by current velocity.

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        lane_coords = self.road.lanes[target_lane_index].local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * (self.TAU_DS + Vehicle.STEERING_TAU)
        lane_future_heading = self.road.lanes[target_lane_index].heading_at(lane_next_coords)
        heading_ref = lane_future_heading - np.arctan(self.KP_S * lane_coords[1] * np.sign(self.velocity) *
                                                      np.exp(-(self.velocity / self.STEERING_VEL_GAIN) ** 2))
        steering = self.KD_S * utils.wrap_to_pi(heading_ref - self.heading) * self.LENGTH / utils.not_zero(
            self.velocity)
        steering = np.clip(steering, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering

    def velocity_control(self, target_velocity):
        """
            Control the velocity of the vehicle.

            Using a simple proportional controller.

        :param target_velocity: the desired velocity
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_velocity - self.velocity)

    def handle_event(self, event):
        """
            Map the pygame keyboard events to actions.

        :param event: the pygame event
        """
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
        A controlled vehicle with a specified discrete range of allowed target velocities.
    """

    SPEED_COUNT = 3  # []
    SPEED_MIN = 20  # [m/s]
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

    def act(self, action=None):
        """
            Perform a high-level action.

            If the action is a velocity change, choose velocity from the allowed discrete range.
            Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        if action == "FASTER":
            self.velocity_index = self.speed_to_index(self.velocity) + 1
        elif action == "SLOWER":
            self.velocity_index = self.speed_to_index(self.velocity) - 1
        else:
            super(MDPVehicle, self).act(action)
            return
        self.velocity_index = np.clip(self.velocity_index, 0, self.SPEED_COUNT - 1)
        self.target_velocity = self.index_to_speed(self.velocity_index)
        super(MDPVehicle, self).act()

    @classmethod
    def index_to_speed(cls, index):
        """
            Convert an index among allowed speeds to its corresponding speed
        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if cls.SPEED_COUNT > 1:
            return cls.SPEED_MIN + index * (cls.SPEED_MAX - cls.SPEED_MIN) / (cls.SPEED_COUNT - 1)
        else:
            return cls.SPEED_MIN

    @classmethod
    def speed_to_index(cls, speed):
        """
            Find the index of the closest speed allowed to a given speed.
        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    def speed_index(self):
        """
            The index of current velocity
        """
        return self.speed_to_index(self.velocity)

    def predict_trajectory(self, actions, action_duration, trajectory_timestep, dt):
        """
            Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states


class IDMVehicle(ControlledVehicle):
    """
        A vehicle using both a longitudinal and a lateral decision policies.

        - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and velocity.
        - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
        """

    IDM_COLOR = Vehicle.BLUE

    # Longitudinal policy parameters
    ACC_MAX = 3.0  # [m/s2]
    BRAKE_ACC = 5.0  # [m/s2]
    VELOCITY_WANTED = 20.0  # [m/s]
    DISTANCE_WANTED = 5.0  # [m]
    TIME_WANTED = 1.0  # [s]
    DELTA = 4.0  # []

    # Lateral policy parameters
    POLITENESS = 0.1  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self, road, position, heading=0, velocity=0, target_lane_index=None, enable_lane_change=True):
        super(IDMVehicle, self).__init__(road, position, heading, velocity, target_lane_index)
        self.enable_lane_change = enable_lane_change
        self.color = self.IDM_COLOR
        self.timer = np.random.random() * self.LANE_CHANGE_DELAY

    def act(self, action=None):
        """
            Execute an action.

            For now, no action is supported because the vehicle takes all decisions
            of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        # Lateral: MOBIL
        if self.enable_lane_change:
            self.change_lane_policy()
        action['steering'] = self.steering_control(self.target_lane_index)

        # Longitudinal: IDM
        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action['acceleration'] = np.clip(action['acceleration'], -self.BRAKE_ACC, self.ACC_MAX)
        super(ControlledVehicle, self).act(action)

    def step(self, dt):
        """
            Step the simulation.

            Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super(IDMVehicle, self).step(dt)

    @classmethod
    def acceleration(cls, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
            Compute an acceleration command with the Intelligent Driver Model.

            The acceleration is chosen so as to:
            - reach a target velocity;
            - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle:
            return 0
        acceleration = cls.ACC_MAX * (
                1 - np.power(ego_vehicle.velocity / utils.not_zero(ego_vehicle.target_velocity), cls.DELTA))
        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle) - ego_vehicle.LENGTH / 2 - front_vehicle.LENGTH / 2
            acceleration -= cls.ACC_MAX * np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    @classmethod
    def desired_gap(cls, ego_vehicle, front_vehicle=None):
        """
            Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :return: the desired distance between the two [m]
        """
        d0 = cls.DISTANCE_WANTED
        tau = cls.TIME_WANTED
        ab = cls.ACC_MAX * cls.BRAKE_ACC
        dv = ego_vehicle.velocity - front_vehicle.velocity
        d_star = d0 + ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab))
        return d_star

    def maximum_velocity(self, front_vehicle=None):
        """
            Compute the maximum allowed velocity to avoid Inevitable Collision States.

            Assume the front vehicle is going to brake at full deceleration and that
            it will be noticed after a given delay, and compute the maximum velocity
            which allows the ego-vehicle to brake enough to avoid the collision.

        :param front_vehicle: the preceding vehicle
        :return: the maximum allowed velocity, and suggested acceleration
        """
        if not front_vehicle:
            return self.target_velocity
        d0 = self.DISTANCE_WANTED
        a0 = self.BRAKE_ACC
        a1 = self.BRAKE_ACC
        tau = self.TIME_WANTED
        d = max(self.lane_distance_to(front_vehicle) - self.LENGTH / 2 - front_vehicle.LENGTH / 2 - d0, 0)
        v1_0 = front_vehicle.velocity
        delta = 4 * (a0 * a1 * tau) ** 2 + 8 * a0 * (a1 ** 2) * d + 4 * a0 * a1 * v1_0 ** 2
        v_max = -a0 * tau + np.sqrt(delta) / (2 * a1)

        # Velocity control
        self.target_velocity = min(self.maximum_velocity(front_vehicle), self.target_velocity)
        acceleration = self.velocity_control(self.target_velocity)

        return v_max, acceleration

    def change_lane_policy(self):
        """
            Decide when to change lane.

            Based on:
            - frequency;
            - closeness of the target lane;
            - MOBIL model.
        """
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # abort it if someone else is already changing into the same lane
            for v in self.road.vehicles:
                if v is not self \
                        and v.lane_index != self.target_lane_index \
                        and isinstance(v, ControlledVehicle) \
                        and v.target_lane_index == self.target_lane_index:
                    d = self.lane_distance_to(v)
                    d_star = self.desired_gap(self, v)
                    if 0 < d < d_star:
                        self.target_lane_index = self.lane_index
                        break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        lane_indexes = [self.lane_index + i for i in [-1, 1] if 0 <= self.lane_index + i < len(self.road.lanes)]
        for lane_index in lane_indexes:
            # Is the candidate lane close enough?
            if not self.road.lanes[lane_index].is_reachable_from(self.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index):
        """
            MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, self.road.lanes[lane_index])
        new_following_a = IDMVehicle.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = IDMVehicle.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Is there an advantage for me and/or my followers to change lane?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_a = IDMVehicle.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
        self_pred_a = IDMVehicle.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        old_following_a = IDMVehicle.acceleration(ego_vehicle=old_following, front_vehicle=self)
        old_following_pred_a = IDMVehicle.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
        jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                         + old_following_pred_a - old_following_a)
        if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
            return False

        # All clear, let's go!
        return True

    def recover_from_stop(self, acceleration):
        """
            If stopped on the wrong lane, try a reversing maneuver.

        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_velocity = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.velocity < stopped_velocity:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.lanes[self.target_lane_index])
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # Reverse
                return -self.ACC_MAX/2
        return acceleration


class LinearVehicle(IDMVehicle):
    ALPHA = 1.0
    BETA_FRONT = 2.0
    BETA_REAR = 0.0
    GAMMA_FRONT = 50.0
    GAMMA_REAR = 0.0
    TIME_WANTED = 2.0

    def __init__(self, road, position, heading=0, velocity=0, target_lane_index=None, enable_lane_change=True):
        super(LinearVehicle, self).__init__(road, position, heading, velocity, target_lane_index, enable_lane_change)
        self.color = Vehicle.PURPLE
        self.target_velocity = self.VELOCITY_WANTED

    @classmethod
    def acceleration(cls, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
            Compute an acceleration command with a Linear Model.

            The acceleration is chosen so as to:
            - reach a target velocity;
            - reach the velocity of the leading (resp following) vehicle, if it is lower (resp higher) than ego's;
            - maintain a minimum safety distance w.r.t the leading vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            Linear vehicle, which is why this method is a class method. This allows a Linear vehicle to
                            reason about other vehicles behaviors even though they may not Linear.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle:
            return 0
        acceleration = cls.ALPHA * (ego_vehicle.target_velocity - ego_vehicle.velocity)
        d_safe = cls.DISTANCE_WANTED + np.max(ego_vehicle.velocity, 0) * cls.TIME_WANTED + ego_vehicle.LENGTH
        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration += cls.BETA_FRONT * min(front_vehicle.velocity - ego_vehicle.velocity, 0) \
                + cls.GAMMA_FRONT * min(d - d_safe, 0)
        if rear_vehicle:
            d = rear_vehicle.lane_distance_to(ego_vehicle)
            acceleration += cls.BETA_REAR * max(rear_vehicle.velocity - ego_vehicle.velocity, 0) \
                + cls.GAMMA_REAR * max(d_safe - d, 0)
        return acceleration


def test():
    from highway.simulation import Simulation
    from highway.road import Road
    road = Road.create_random_road(lanes_count=2, lane_width=4.0, vehicles_count=30, vehicles_type=LinearVehicle)
    sim = Simulation(road, ego_vehicle_type=ControlledVehicle)

    while not sim.done:
        sim.process()
    sim.quit()


if __name__ == '__main__':
    test()
