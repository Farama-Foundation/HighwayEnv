from __future__ import division, print_function
import numpy as np
import copy
from highway import utils
from highway.vehicle.dynamics import Vehicle


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

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.velocity,
                vehicle.target_lane_index, vehicle.target_velocity)
        return v

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


class MDPVehicle(ControlledVehicle):
    """
        A controlled vehicle with a specified discrete range of allowed target velocities.
    """

    SPEED_COUNT = 3  # []
    SPEED_MIN = 20  # [m/s]
    SPEED_MAX = 30  # [m/s]

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
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
