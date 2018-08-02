import copy
import numpy as np

from highway_env.vehicle.behavior import LinearVehicle


class VehicleInterval(object):
    def __init__(self, position, velocity, heading):
        self.position = position
        self.velocity = velocity
        self.heading = heading


class IntervalVehicle(LinearVehicle):
    """
        Observer for the interval-membership of a LinearVehicle under parameter uncertainty.

        The model trajectory is stored in a model_vehicle, and the lower and upper bounds of the states are stored
        in a min_vehicle and max_vehicle. Note that these vehicles do not follow a proper Vehicle dynamics, and
        are only used for storage of the bounds.
    """
    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 enable_lane_change=True,
                 timer=None,
                 theta_a_i=None,
                 theta_b_i=None):
        """
        :param theta_a_i: The interval of possible acceleration parameters
        :param theta_b_i: The interval of possible steering parameters
        """
        super(IntervalVehicle, self).__init__(road,
                                              position,
                                              heading,
                                              velocity,
                                              target_lane_index,
                                              target_velocity,
                                              enable_lane_change,
                                              timer)
        theta_a = np.array(LinearVehicle.ACCELERATION_PARAMETERS)
        theta_b = np.array(LinearVehicle.STEERING_GAIN)
        self.theta_a_i = theta_a_i if theta_a_i is not None else np.array([0.5*theta_a, 2*theta_a])
        self.theta_b_i = theta_b_i if theta_b_i is not None else np.array([0.5*theta_b, 2*theta_b])

        self.interval_observer = VehicleInterval(np.array([self.position, self.position], dtype=float),
                                                 np.array([self.velocity, self.velocity], dtype=float),
                                                 np.array([self.heading, self.heading], dtype=float))
        self.trajectory = []
        self.observer_trajectory = []

    @classmethod
    def create_from(cls, vehicle):
        timer = None if not hasattr(vehicle, 'timer') else vehicle.timer
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity,
                timer=timer, theta_a_i=None, theta_b_i=None)
        if isinstance(vehicle, IntervalVehicle):
            v.interval_observer = copy.deepcopy(vehicle.interval_observer)
        return v

    def step(self, dt):
        # self.observer_step(dt)
        self.partial_step(dt)
        self.store_trajectories()
        super(IntervalVehicle, self).step(dt)

    def observer_step(self, dt, lane_change_model="model"):
        """
            Step the interval observer dynamics
        :param dt: timestep [s]
        :param lane_change_model: - model: assume that the vehicle will follow the lane of its model behaviour.
                                  - all: assume that any lane change decision is possible at any timestep
                                  - right: assume that a right lane change decision is possible at any timestep
        """
        # Input state intervals
        position_i = self.interval_observer.position
        v_i = self.interval_observer.velocity
        psi_i = self.interval_observer.heading

        # Features interval
        # TODO: For now, we assume the front vehicle follows the models' front vehicle
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if front_vehicle:
            if isinstance(front_vehicle, IntervalVehicle):
                # Use interval from the observer estimate of the front vehicle
                front_observer = front_vehicle.interval_observer
            else:
                # The front vehicle trajectory interval is not being estimated, so it should be considered as certain.
                # We use a new observer created from that current vehicle state, which will have full certainty.
                front_observer = IntervalVehicle.create_from(front_vehicle).interval_observer
        else:
            front_observer = None

        # Acceleration features
        phi_a_i = np.zeros((2, 3))
        phi_a_i[:, 0] = [0, 0]
        if front_observer:
            phi_a_i[:, 1] = IntervalVehicle.interval_negative_part(
                IntervalVehicle.intervals_diff(front_observer.velocity, v_i))
            # TODO: Use proper lane distance instead of X difference
            d_i = IntervalVehicle.intervals_diff(front_observer.position[:, 0], position_i[:, 0])
            d_safe_i = self.DISTANCE_WANTED + self.LENGTH + self.TIME_WANTED * v_i
            phi_a_i[:, 2] = IntervalVehicle.interval_negative_part(IntervalVehicle.intervals_diff(d_i, d_safe_i))

        # Steering features
        phi_b_i = None
        if lane_change_model == "model":
            lanes = [self.target_lane_index]
        elif lane_change_model == "all":
            lanes = range(len(self.road.lanes))
        elif lane_change_model == "right":
            lanes = range(self.target_lane_index, len(self.road.lanes))
        for lane_index in lanes:
            lane_coords = self.road.lanes[lane_index].local_coordinates(self.position)
            lane_y = self.position[1] - lane_coords[1]
            lane_psi = self.road.lanes[self.target_lane_index].heading_at(lane_coords[0])
            i_v_i = 1/np.flip(v_i, 0)
            phi_b_i_lane = np.transpose(np.array(
                [IntervalVehicle.intervals_product((lane_y - np.flip(position_i[:, 1], 0)), i_v_i),
                 [0, 0]]))
            # Union of candidate feature intervals
            if phi_b_i is None:
                phi_b_i = phi_b_i_lane
            else:
                phi_b_i[0] = np.minimum(phi_b_i[0], phi_b_i_lane[0])
                phi_b_i[1] = np.maximum(phi_b_i[1], phi_b_i_lane[1])

        # Commands interval
        a_i = IntervalVehicle.intervals_product(self.theta_a_i, phi_a_i)
        b_i = IntervalVehicle.intervals_product(self.theta_b_i, phi_b_i)

        # Velocities interval
        keep_stability = True
        if keep_stability:
            dv_i = IntervalVehicle.integrator_interval(v_i - self.target_velocity, self.theta_a_i[:, 0])
        else:
            dv_i = IntervalVehicle.intervals_product(self.theta_b_i[:, 1], self.target_velocity - np.flip(v_i, 0))
        dv_i += a_i
        if keep_stability:
            d_psi_i = IntervalVehicle.integrator_interval(psi_i - lane_psi, self.theta_b_i[:, 1])
        else:
            d_psi_i = IntervalVehicle.intervals_product(self.theta_b_i[:, 1], lane_psi - np.flip(psi_i, 0))
        d_psi_i += b_i

        # Position interval
        cos_i = [-1 if psi_i[0] <= np.pi <= psi_i[1] else min(map(np.cos, psi_i)),
                 1 if psi_i[0] <= 0 <= psi_i[1] else max(map(np.cos, psi_i))]
        sin_i = [-1 if psi_i[0] <= -np.pi/2 <= psi_i[1] else min(map(np.sin, psi_i)),
                 1 if psi_i[0] <= np.pi/2 <= psi_i[1] else max(map(np.sin, psi_i))]
        dx_i = IntervalVehicle.intervals_product(v_i, cos_i)
        dy_i = IntervalVehicle.intervals_product(v_i, sin_i)

        # Interval dynamics integration
        self.interval_observer.velocity += dv_i * dt
        self.interval_observer.heading += d_psi_i * dt
        self.interval_observer.position[:, 0] += dx_i * dt
        self.interval_observer.position[:, 1] += dy_i * dt

    def partial_step(self, dt, alpha=0):
        """
            Step the boundary parts of the current state interval

            1. Split x_i(t) into two upper and lower intervals x_i_-(t) and x_i_+(t)
            2. Propagate their observer dynamics x_i_-(t+dt) and x_i_+(t+dt)
            3. Merge the resulting intervals together to x_i(t+dt).
        :param dt: timestep [s]
        :param alpha: ratio of the full interval that defines the boundaries
        """
        # 1. Split x_i(t) into two upper and lower intervals x_i_-(t) and x_i_+(t)
        o = self.interval_observer
        v_minus = IntervalVehicle.create_from(self)
        v_minus.interval_observer.position[1, :] = (1 - alpha) * o.position[0, :] + alpha * o.position[1, :]
        v_minus.interval_observer.velocity[1] = (1 - alpha) * o.velocity[0] + alpha * o.velocity[1]
        v_minus.interval_observer.heading[1] = (1 - alpha) * o.heading[0] + alpha * o.heading[1]
        v_plus = IntervalVehicle.create_from(self)
        v_plus.interval_observer.position[0, :] = alpha * o.position[0, :] + (1 - alpha) * o.position[1, :]
        v_plus.interval_observer.velocity[0] = alpha * o.velocity[0] + (1 - alpha) * o.velocity[1]
        v_plus.interval_observer.heading[0] = alpha * o.heading[0] + (1 - alpha) * o.heading[1]
        # 2. Propagate their observer dynamics x_i_-(t+dt) and x_i_+(t+dt)
        v_minus.road = copy.copy(v_minus.road)
        v_minus.road.vehicles = [v if v is not self else v_minus for v in v_minus.road.vehicles]
        v_plus.road = copy.copy(v_plus.road)
        v_plus.road.vehicles = [v if v is not self else v_plus for v in v_plus.road.vehicles]
        v_minus.observer_step(dt)
        v_plus.observer_step(dt)
        # 3. Merge the resulting intervals together to x_i(t+dt).
        self.interval_observer.position = np.array([v_minus.interval_observer.position[0], v_plus.interval_observer.position[1]])
        self.interval_observer.velocity = np.array([v_minus.interval_observer.velocity[0], v_plus.interval_observer.velocity[1]])
        self.interval_observer.heading = np.array([v_minus.interval_observer.heading[0], v_plus.interval_observer.heading[1]])

    def store_trajectories(self):
        """
            Store the current model, min and max states to a trajectory list
        """
        self.trajectory.append(LinearVehicle.create_from(self))
        self.observer_trajectory.append(copy.deepcopy(self.interval_observer))

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

    @staticmethod
    def integrator_interval(x, k):
        """
            Compute the interval of an integrator system: dx = -k*x
        :param x: state interval
        :param k: gain interval, must be positive
        :return: interval for dx
        """

        if x[0] >= 0:
            interval_gain = np.flip(-k, 0)
        elif x[1] <= 0:
            interval_gain = -k
        else:
            interval_gain = -np.array([k[0], k[0]])
        return interval_gain*x  # Note: no flip of x, contrary to using intervals_product(k,interval_minus(x))

