from __future__ import division, print_function
import numpy as np
import copy
from highway_env import utils
from highway_env.vehicle.dynamics import Vehicle


class ControlledVehicle(Vehicle):
    """
        A vehicle piloted by two low-level controller, allowing high-level actions
        such as cruise control and lane changes.

        - The longitudinal controller is a velocity controller;
        - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    PURSUIT_TAU = 1.5*TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.5  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]

    DELTA_VELOCITY = 5  # [m/s]

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None):
        super(ControlledVehicle, self).__init__(road, position, heading, velocity)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_velocity = target_velocity or self.velocity
        self.route = route

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity,
                route=vehicle.route)
        return v

    def plan_route_to(self, destination):
        """
            Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        path = self.road.network.shortest_path(self.lane_index[1], destination)
        if path:
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action=None):
        """
            Perform a high-level action to change the desired lane or velocity.

            - If a high-level action is provided, update the target velocity and lane;
            - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.follow_road()
        if action == "FASTER":
            self.target_velocity += self.DELTA_VELOCITY
        elif action == "SLOWER":
            self.target_velocity -= self.DELTA_VELOCITY
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index

        action = {'steering': self.steering_control(self.target_lane_index),
                  'acceleration': self.velocity_control(self.target_velocity)}
        super(ControlledVehicle, self).act(action)

    def follow_road(self):
        """
           At the end of a lane, automatically switch to a next one.
        """
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)

    def steering_control(self, target_lane_index):
        """
            Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral velocity command
        2. Lateral velocity command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * self.PURSUIT_TAU
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_velocity_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral velocity to heading
        heading_command = np.arcsin(np.clip(lateral_velocity_command/utils.not_zero(self.velocity), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(self.velocity) * heading_rate_command)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering_angle

    def velocity_control(self, target_velocity):
        """
            Control the velocity of the vehicle.

            Using a simple proportional controller.

        :param target_velocity: the desired velocity
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_velocity - self.velocity)

    def set_route_at_intersection(self, _to):
        """
            Set the road to be followed at the next intersection.
            Erase current planned route.
        :param _to: index of the road to follow at next intersection, in the road network
        """

        if not self.route:
            return
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return
        next_destinations_from = list(next_destinations.keys())
        if _to == "random":
            _to = self.road.np_random.randint(0, len(next_destinations_from))
        next_index = _to % len(next_destinations_from)
        self.route = self.route[0:index+1] + \
                     [(self.route[index][1], next_destinations_from[next_index], self.route[index][2])]

    def predict_trajectory_constant_velocity(self, times):
        """
            Predict the future positions of the vehicle along its planned route, under constant velocity
        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        return zip(*[self.road.network.position_heading_along_route(route, coordinates[0] + self.velocity * t, 0)
                     for t in times])


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
                 target_velocity=None,
                 route=None):
        super(MDPVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route)
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
