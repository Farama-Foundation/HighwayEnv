from __future__ import division, print_function
import numpy as np
from highway_env.vehicle.control import ControlledVehicle
from highway_env import utils


class IDMVehicle(ControlledVehicle):
    """
        A vehicle using both a longitudinal and a lateral decision policies.

        - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and velocity.
        - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
        """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    COMFORT_ACC_MAX = 3.0  # [m/s2]
    COMFORT_ACC_MIN = -5.0  # [m/s2]
    VELOCITY_WANTED = 20.0  # [m/s]
    DISTANCE_WANTED = 5.0  # [m]
    TIME_WANTED = 1.5  # [s]
    DELTA = 4.0  # []

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 enable_lane_change=True,
                 timer=None):
        super(IDMVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        timer = None if not hasattr(vehicle, 'timer') else vehicle.timer
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity,
                timer=timer)
        return v

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
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
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
        acceleration = cls.COMFORT_ACC_MAX * (
                1 - np.power(ego_vehicle.velocity / utils.not_zero(ego_vehicle.target_velocity), cls.DELTA))
        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= cls.COMFORT_ACC_MAX * \
                np.power(cls.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    @classmethod
    def desired_gap(cls, ego_vehicle, front_vehicle=None):
        """
            Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :return: the desired distance between the two [m]
        """
        d0 = cls.DISTANCE_WANTED + ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2
        tau = cls.TIME_WANTED
        ab = -cls.COMFORT_ACC_MAX * cls.COMFORT_ACC_MIN
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
        a0 = self.COMFORT_ACC_MIN
        a1 = self.COMFORT_ACC_MIN
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
        self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
        old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
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
                return -self.COMFORT_ACC_MAX / 2
        return acceleration


class LinearVehicle(IDMVehicle):
    PARAMETERS = [0.3, 0.14, 0.8]
    TIME_WANTED = 2.0

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 enable_lane_change=True,
                 timer=None):
        super(LinearVehicle, self).__init__(road,
                                            position,
                                            heading,
                                            velocity,
                                            target_lane_index,
                                            target_velocity,
                                            enable_lane_change,
                                            timer)

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
        return np.dot(cls.PARAMETERS, cls.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle))

    @classmethod
    def acceleration_features(cls, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = ego_vehicle.target_velocity - ego_vehicle.velocity
            d_safe = cls.DISTANCE_WANTED + np.max(ego_vehicle.velocity, 0) * cls.TIME_WANTED + ego_vehicle.LENGTH
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.velocity - ego_vehicle.velocity, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])


class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                  MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                  0.5]


class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                  MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                  2.0]


class IntervalObserver(object):
    """
        Observer for the interval-membership of a LinearVehicle under parameter uncertainty.

        The model trajectory is stored in a model_vehicle, and the lower and upper bounds of the states are stored
        in a min_vehicle and max_vehicle. Note that these vehicles do not follow a proper Vehicle dynamics, and
        are only used for storage of the bounds.
    """
    def __init__(self,
                 road_observer,
                 vehicle,
                 theta_a=LinearVehicle.PARAMETERS,
                 theta_b=ControlledVehicle.STEERING_GAIN,
                 theta_a_i=None,
                 theta_b_i=None):
        """
        :param RoadObserver road_observer: The observer of the nearby traffic
        :param vehicle: The vehicle whose trajectory interval is estimated
        :param theta_a: An acceleration parameter estimate
        :param theta_b: A steering parameter estimate
        :param theta_a_i: The interval of possible acceleration parameters
        :param theta_b_i: The interval of possible steering parameters
        """
        self.road_observer = road_observer
        self.vehicle = vehicle
        self.theta_a = np.array(theta_a)
        self.theta_b = np.array(theta_b)
        self.theta_a_i = theta_a_i if theta_a_i is not None else np.array([0.5 * self.theta_a, 2 * self.theta_a])
        self.theta_b_i = theta_b_i if theta_b_i is not None else np.array([0.5 * self.theta_b, 2 * self.theta_b])

        self.model_vehicle = None
        self.min_vehicle = None
        self.max_vehicle = None
        self.model_traj, self.min_traj, self.max_traj = [], [], []

    def duplicate(self):
        new_observer = IntervalObserver(self.road_observer,
                         self.vehicle,
                         theta_a=self.theta_a,
                         theta_b=self.theta_b,
                         theta_a_i=self.theta_a_i,
                         theta_b_i=self.theta_b_i)
        new_observer.model_vehicle = LinearVehicle.create_from(self.model_vehicle)
        new_observer.min_vehicle = LinearVehicle.create_from(self.min_vehicle)
        new_observer.max_vehicle = LinearVehicle.create_from(self.max_vehicle)
        return new_observer

    def step(self, dt):
        self.partial_step(dt)

    def observer_step(self, dt):
        """
            Step the interval observer dynamics
        :param dt: timestep [s]
        """
        # Input state intervals
        x_i = [self.min_vehicle.position[0], self.max_vehicle.position[0]]
        y_i = np.array([self.min_vehicle.position[1], self.max_vehicle.position[1]])
        v_i = [self.min_vehicle.velocity, self.max_vehicle.velocity]
        psi_i = [self.min_vehicle.heading, self.max_vehicle.heading]

        # Features interval
        # TODO: For now, we assume the current front vehicle will stay the same at short term
        front_vehicle, _ = self.vehicle.road.neighbour_vehicles(self.vehicle)
        front_observer = self.road_observer.observers[front_vehicle] if front_vehicle else None
        front_model_vehicle = front_observer.model_vehicle if front_vehicle else None
        # Acceleration features
        phi_a = self.model_vehicle.acceleration_features(ego_vehicle=self.model_vehicle,
                                                         front_vehicle=front_model_vehicle)
        phi_a_i = np.zeros((2, 3))
        phi_a_i[:, 0] = IntervalObserver.intervals_diff([self.model_vehicle.target_velocity]*2, v_i)
        if front_observer:
            v_f_i = [front_observer.min_vehicle.velocity, front_observer.max_vehicle.velocity]
            phi_a_i[:, 1] = IntervalObserver.interval_negative_part(IntervalObserver.intervals_diff(v_f_i, v_i))
            # TODO: Use proper lane distance instead of X difference
            xf_i = [front_observer.min_vehicle.position[0], front_observer.max_vehicle.position[0]]
            d_i = IntervalObserver.intervals_diff(xf_i, x_i)
            d_safe_i = LinearVehicle.DISTANCE_WANTED + self.vehicle.LENGTH + LinearVehicle.TIME_WANTED * np.array(v_i)
            phi_a_i[:, 2] = IntervalObserver.interval_negative_part(IntervalObserver.intervals_diff(d_i, d_safe_i))

        # Steering features
        # TODO: For now, we assume the current target lane will stay the same at short term
        phi_b = self.model_vehicle.steering_features(self.vehicle.target_lane_index)
        lane_coords = self.vehicle.road.lanes[self.vehicle.target_lane_index].local_coordinates(self.vehicle.position)
        lane_y = self.vehicle.position[1] - lane_coords[1]
        lane_psi = self.vehicle.road.lanes[self.vehicle.target_lane_index].heading_at(lane_coords[0])
        i_v_i = 1/np.flip(v_i, 0)
        phi_b_i = np.transpose(np.array(
            [IntervalObserver.intervals_product((lane_y - np.flip(y_i, 0)) * self.vehicle.LENGTH, i_v_i ** 2),
             IntervalObserver.intervals_product((lane_psi - np.flip(psi_i, 0)) * self.vehicle.LENGTH, i_v_i)]))

        # Commands interval
        a_i = IntervalObserver.intervals_product(self.theta_a_i, phi_a_i)
        b_i = IntervalObserver.intervals_product(self.theta_b_i, phi_b_i)

        # Velocities interval
        dv_i = a_i
        tan_b_i = [np.tan(b_i[0])/self.vehicle.LENGTH, np.tan(b_i[1])/self.vehicle.LENGTH]
        d_psi_i = IntervalObserver.intervals_product(v_i, tan_b_i)

        # Position interval
        cos_i = [-1 if psi_i[0] <= np.pi <= psi_i[1] else min(map(np.cos, psi_i)),
                 1 if psi_i[0] <= 0 <= psi_i[1] else max(map(np.cos, psi_i))]
        sin_i = [-1 if psi_i[0] <= -np.pi/2 <= psi_i[1] else min(map(np.sin, psi_i)),
                 1 if psi_i[0] <= np.pi/2 <= psi_i[1] else max(map(np.sin, psi_i))]
        dx_i = IntervalObserver.intervals_product(v_i, cos_i)
        dy_i = IntervalObserver.intervals_product(v_i, sin_i)

        # Interval dynamics integration
        self.min_vehicle.action['acceleration'] = a_i[0]
        self.min_vehicle.action['steering'] = b_i[0]
        self.min_vehicle.velocity += dv_i[0]*dt
        self.min_vehicle.heading += d_psi_i[0]*dt
        self.min_vehicle.position[0] += dx_i[0]*dt
        self.min_vehicle.position[1] += dy_i[0]*dt
        self.max_vehicle.action['acceleration'] = a_i[1]
        self.max_vehicle.action['steering'] = b_i[1]
        self.max_vehicle.velocity += dv_i[1] * dt
        self.max_vehicle.heading += d_psi_i[1] * dt
        self.max_vehicle.position[0] += dx_i[1] * dt
        self.max_vehicle.position[1] += dy_i[1]*dt
        self.model_vehicle.action['acceleration'] = np.dot(self.theta_a, phi_a)
        self.model_vehicle.action['steering'] = np.dot(self.theta_b, phi_b)
        self.model_vehicle.position += self.model_vehicle.velocity*np.array([np.cos(self.model_vehicle.heading),
                                                                             np.sin(self.model_vehicle.heading)])*dt
        self.model_vehicle.velocity += self.model_vehicle.action['acceleration']*dt
        self.model_vehicle.heading += self.model_vehicle.velocity/self.vehicle.LENGTH * \
                                      np.tan(self.model_vehicle.action['steering'])*dt

        # Store trajectories
        self.store_trajectories()

    def partial_step(self, dt, alpha=0):
        observer_minus = self.duplicate()
        observer_minus.max_vehicle.position = (1 - alpha) * observer_minus.min_vehicle.position + \
                                              alpha * observer_minus.max_vehicle.position
        observer_minus.max_vehicle.velocity = (1 - alpha) * observer_minus.min_vehicle.velocity + \
                                              alpha * observer_minus.max_vehicle.velocity
        observer_minus.max_vehicle.heading = (1 - alpha) * observer_minus.min_vehicle.heading + \
                                             alpha * observer_minus.max_vehicle.heading
        observer_plus = self.duplicate()
        observer_plus.min_vehicle.position = alpha * observer_plus.min_vehicle.position + \
                                             (1 - alpha) * observer_plus.max_vehicle.position
        observer_plus.min_vehicle.velocity = alpha * observer_plus.min_vehicle.velocity + \
                                             (1 - alpha) * observer_plus.max_vehicle.velocity
        observer_plus.min_vehicle.heading = alpha * observer_plus.min_vehicle.heading + \
                                            (1 - alpha) * observer_plus.max_vehicle.heading
        observer_minus.observer_step(dt)
        observer_plus.observer_step(dt)
        self.observer_step(dt)  # In order to step the model vehicle
        self.min_vehicle = observer_minus.min_vehicle
        self.max_vehicle = observer_plus.max_vehicle
        self.store_trajectories()

    def store_trajectories(self):
        self.model_traj.append(LinearVehicle.create_from(self.model_vehicle))
        self.min_traj.append(LinearVehicle.create_from(self.min_vehicle))
        self.max_traj.append(LinearVehicle.create_from(self.max_vehicle))

    def reset(self):
        """
            Reset the model and bounds estimates on current vehicle state, and clear trajectories
        """
        self.model_traj, self.min_traj, self.max_traj = [], [], []
        self.model_vehicle = LinearVehicle.create_from(self.vehicle)
        self.min_vehicle = LinearVehicle.create_from(self.vehicle)
        self.max_vehicle = LinearVehicle.create_from(self.vehicle)

    @staticmethod
    def intervals_product(a, b):
        """
            Compute the product of two intervals
        :param a: interval [a_min, a_max]
        :param b: interval [b_min, b_max]
        :return: the interval of their product ab
        """
        p = lambda x: np.maximum(x, 0)
        n = lambda x: np.maximum(-x, 0)
        return np.array(
            [np.dot(p(a[0]), p(b[0])) - np.dot(p(a[1]), n(b[0])) - np.dot(n(a[0]), p(b[1])) + np.dot(n(a[1]), n(b[1])),
             np.dot(p(a[1]), p(b[1])) - np.dot(p(a[0]), n(b[1])) - np.dot(n(a[1]), p(b[0])) + np.dot(n(a[0]), n(b[0]))])

    @staticmethod
    def intervals_diff(a, b):
        """
            Compute the difference of two intervals
        :param a: interval [a_min, a_max]
        :param b: interval [b_min, b_max]
        :return: the interval of their difference a - b
        """
        return np.array([a[0] - b[1], a[1] - b[0]])

    @staticmethod
    def interval_negative_part(a):
        """
            Compute the negative part of an interval
        :param a: interval [a_min, a_max]
        :return: the interval of its negative part min(a, 0)
        """
        return np.minimum(a, 0)

    def get_trajectories(self):
        return self.model_traj, self.min_traj, self.max_traj


class RoadObserver(object):
    def __init__(self, road):
        self.road = road
        self.observers = {vehicle: IntervalObserver(self, vehicle)
                          for vehicle in road.vehicles if isinstance(vehicle, ControlledVehicle)}

    def step(self, dt):
        for observer in self.observers.values():
            observer.step(dt)

    def compute_trajectories(self, n, dt):
        """
            Compute for each vehicle model trajectory and corresponding state intervals
            :param n: number of steps in the trajectory
            :param dt: timestep [s]
        """
        for observer in self.observers.values():
            observer.reset()
        for _ in range(n):
            self.step(dt)
