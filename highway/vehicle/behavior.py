from __future__ import division, print_function
import numpy as np
from highway.vehicle.control import ControlledVehicle
from highway import utils


class IDMVehicle(ControlledVehicle):
    """
        A vehicle using both a longitudinal and a lateral decision policies.

        - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and velocity.
        - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
        """

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
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
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
