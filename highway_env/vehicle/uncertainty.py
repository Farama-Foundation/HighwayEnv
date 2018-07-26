import numpy as np

from highway_env.vehicle.behavior import LinearVehicle
from highway_env.vehicle.control import ControlledVehicle


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
                 theta_a=LinearVehicle.ACCELERATION_PARAMETERS,
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

    def copy(self):
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
        self.store_trajectories()

    def observer_step(self, dt, assume_constant_lane=True):
        """
            Step the interval observer dynamics
        :param dt: timestep [s]
        :param assume_constant_lane: If true, assume that the vehicle will stay on its current lane.
                                    Else, assume that a lane change decision is possible at any timestep
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
        phi_b = self.model_vehicle.steering_features(self.vehicle.target_lane_index)
        phi_b_i = None
        lanes = [self.vehicle.target_lane_index] if assume_constant_lane else range(len(self.road_observer.road.lanes))
        for lane_index in lanes:
            lane_coords = self.vehicle.road.lanes[lane_index].local_coordinates(self.vehicle.position)
            lane_y = self.vehicle.position[1] - lane_coords[1]
            lane_psi = self.vehicle.road.lanes[self.vehicle.target_lane_index].heading_at(lane_coords[0])
            i_v_i = 1/np.flip(v_i, 0)
            phi_b_i_lane = np.transpose(np.array(
                [IntervalObserver.intervals_product((lane_y - np.flip(y_i, 0)) * self.vehicle.LENGTH, i_v_i ** 2),
                 IntervalObserver.intervals_product((lane_psi - np.flip(psi_i, 0)) * self.vehicle.LENGTH, i_v_i)]))
            # Union of candidate feature intervals
            if phi_b_i is None:
                phi_b_i = phi_b_i_lane
            else:
                phi_b_i[0] = np.minimum(phi_b_i[0], phi_b_i_lane[0])
                phi_b_i[1] = np.maximum(phi_b_i[1], phi_b_i_lane[1])

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

    def partial_step(self, dt, alpha=0):
        """
            Step the boundary parts of the current state interval

            1. Split x_i(t) into two upper and lower intervals x_i_-(t) and x_i_+(t)
            2. Propagate their observer dynamics x_i_-(t+dt) and x_i_+(t+dt)
            3. Merge the resulting intervals together to x_i(t+dt).
        :param dt: timestep [s]
        :param alpha: ratio of the full interval that defines the boundaries
        """
        observer_minus = self.copy()
        observer_minus.max_vehicle.position = (1 - alpha) * observer_minus.min_vehicle.position + \
                                              alpha * observer_minus.max_vehicle.position
        observer_minus.max_vehicle.velocity = (1 - alpha) * observer_minus.min_vehicle.velocity + \
                                              alpha * observer_minus.max_vehicle.velocity
        observer_minus.max_vehicle.heading = (1 - alpha) * observer_minus.min_vehicle.heading + \
                                             alpha * observer_minus.max_vehicle.heading
        observer_plus = self.copy()
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

    def store_trajectories(self):
        """
            Store the current model, min and max states to a trajectory list
        """
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
