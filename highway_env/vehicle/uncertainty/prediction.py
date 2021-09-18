import copy
from typing import List, Tuple, Callable, Union, TYPE_CHECKING
import numpy as np

from highway_env import utils
from highway_env.interval import polytope, vector_interval_section, integrator_interval, \
    interval_negative_part, intervals_diff, intervals_product, LPV, interval_absolute_to_local, \
    interval_local_to_absolute
from highway_env.road.road import Route, LaneIndex, Road
from highway_env.utils import Vector
from highway_env.vehicle.behavior import LinearVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle

if TYPE_CHECKING:
    from highway_env.vehicle.objects import RoadObject

Polytope = Tuple[np.ndarray, List[np.ndarray]]


class IntervalVehicle(LinearVehicle):

    """
    Estimator for the interval-membership of a LinearVehicle under parameter uncertainty.

    The model trajectory is stored in a model_vehicle, and the lower and upper bounds of the states are stored
    in a min_vehicle and max_vehicle. Note that these vehicles do not follow a proper Vehicle dynamics, and
    are only used for storage of the bounds.
    """

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 theta_a_i: List[List[float]] = None,
                 theta_b_i: List[List[float]] = None,
                 data: dict = None) -> None:
        """
        :param theta_a_i: The interval of possible acceleration parameters
        :param theta_b_i: The interval of possible steering parameters
        """
        super().__init__(road,
                         position,
                         heading,
                         speed,
                         target_lane_index,
                         target_speed,
                         route,
                         enable_lane_change,
                         timer)
        self.theta_a_i = theta_a_i if theta_a_i is not None else LinearVehicle.ACCELERATION_RANGE
        self.theta_b_i = theta_b_i if theta_b_i is not None else LinearVehicle.STEERING_RANGE
        self.data = data
        self.interval = VehicleInterval(self)
        self.trajectory = []
        self.interval_trajectory = []
        self.longitudinal_lpv, self.lateral_lpv = None, None
        self.previous_target_lane_index = self.target_lane_index

    @classmethod
    def create_from(cls, vehicle: LinearVehicle) -> "IntervalVehicle":
        v = cls(vehicle.road,
                vehicle.position,
                heading=vehicle.heading,
                speed=vehicle.speed,
                target_lane_index=getattr(vehicle, 'target_lane_index', None),
                target_speed=getattr(vehicle, 'target_speed', None),
                route=getattr(vehicle, 'route', None),
                timer=getattr(vehicle, 'timer', None),
                theta_a_i=getattr(vehicle, 'theta_a_i', None),
                theta_b_i=getattr(vehicle, 'theta_b_i', None),
                data=getattr(vehicle, "data", None))
        return v

    def step(self, dt: float, mode: str = "partial") -> None:
        self.store_trajectories()
        if self.crashed:
            self.interval = VehicleInterval(self)
        else:
            if mode == "partial":
                # self.observer_step(dt)
                self.partial_observer_step(dt)
            elif mode == "predictor":
                self.predictor_step(dt)
        super().step(dt)

    def observer_step(self, dt: float) -> None:
        """
        Step the interval observer dynamics

        :param dt: timestep [s]
        """
        # Input state intervals
        position_i = self.interval.position
        v_i = self.interval.speed
        psi_i = self.interval.heading

        # Features interval
        front_interval = self.get_front_interval()

        # Acceleration features
        phi_a_i = np.zeros((2, 3))
        phi_a_i[:, 0] = [0, 0]
        if front_interval:
            phi_a_i[:, 1] = interval_negative_part(
                intervals_diff(front_interval.speed, v_i))
            # Lane distance interval
            lane_psi = self.lane.heading_at(self.lane.local_coordinates(self.position)[0])
            lane_direction = [np.cos(lane_psi), np.sin(lane_psi)]
            diff_i = intervals_diff(front_interval.position, position_i)
            d_i = vector_interval_section(diff_i, lane_direction)

            d_safe_i = self.DISTANCE_WANTED + self.TIME_WANTED * v_i
            phi_a_i[:, 2] = interval_negative_part(intervals_diff(d_i, d_safe_i))

        # Steering features
        phi_b_i = None
        lanes = self.get_followed_lanes()
        for lane_index in lanes:
            lane = self.road.network.get_lane(lane_index)
            longitudinal_pursuit = lane.local_coordinates(self.position)[0] + self.speed * self.TAU_PURSUIT
            lane_psi = lane.heading_at(longitudinal_pursuit)
            _, lateral_i = interval_absolute_to_local(position_i, lane)
            lateral_i = -np.flip(lateral_i)
            i_v_i = 1/np.flip(v_i, 0)
            phi_b_i_lane = np.transpose(np.array([
                [0, 0],
                intervals_product(lateral_i, i_v_i)]))
            # Union of candidate feature intervals
            if phi_b_i is None:
                phi_b_i = phi_b_i_lane
            else:
                phi_b_i[0] = np.minimum(phi_b_i[0], phi_b_i_lane[0])
                phi_b_i[1] = np.maximum(phi_b_i[1], phi_b_i_lane[1])

        # Commands interval
        a_i = intervals_product(self.theta_a_i, phi_a_i)
        b_i = intervals_product(self.theta_b_i, phi_b_i)

        # Speeds interval
        keep_stability = False
        if keep_stability:
            dv_i = integrator_interval(v_i - self.target_speed, self.theta_a_i[:, 0])
        else:
            dv_i = intervals_product(self.theta_a_i[:, 0], self.target_speed - np.flip(v_i, 0))
        dv_i += a_i
        dv_i = np.clip(dv_i, -self.ACC_MAX, self.ACC_MAX)
        keep_stability = True
        if keep_stability:
            delta_psi = list(map(utils.wrap_to_pi, psi_i - lane_psi))
            d_psi_i = integrator_interval(delta_psi, self.theta_b_i[:, 0])
        else:
            d_psi_i = intervals_product(self.theta_b_i[:, 0], lane_psi - np.flip(psi_i, 0))
        d_psi_i += b_i

        # Position interval
        cos_i = [-1 if psi_i[0] <= np.pi <= psi_i[1] else min(map(np.cos, psi_i)),
                 1 if psi_i[0] <= 0 <= psi_i[1] else max(map(np.cos, psi_i))]
        sin_i = [-1 if psi_i[0] <= -np.pi/2 <= psi_i[1] else min(map(np.sin, psi_i)),
                 1 if psi_i[0] <= np.pi/2 <= psi_i[1] else max(map(np.sin, psi_i))]
        dx_i = intervals_product(v_i, cos_i)
        dy_i = intervals_product(v_i, sin_i)

        # Interval dynamics integration
        self.interval.speed += dv_i * dt
        self.interval.heading += d_psi_i * dt
        self.interval.position[:, 0] += dx_i * dt
        self.interval.position[:, 1] += dy_i * dt

        # Add noise
        noise = 0.3
        self.interval.position[:, 0] += noise * dt * np.array([-1, 1])
        self.interval.position[:, 1] += noise * dt * np.array([-1, 1])
        self.interval.heading += noise * dt * np.array([-1, 1])

    def predictor_step(self, dt: float) -> None:
        """
        Step the interval predictor dynamics

        :param dt: timestep [s]
        """
        # Create longitudinal and lateral LPVs
        self.predictor_init()

        # Detect lane change and update intervals of local coordinates with the new frame
        if self.target_lane_index != self.previous_target_lane_index:
            position_i = self.interval.position
            target_lane = self.road.network.get_lane(self.target_lane_index)
            previous_target_lane = self.road.network.get_lane(self.previous_target_lane_index)
            longi_i, lat_i = interval_absolute_to_local(position_i, target_lane)
            psi_i = self.interval.heading + \
                    target_lane.heading_at(longi_i.mean()) - previous_target_lane.heading_at(longi_i.mean())
            x_i_local_unrotated = np.transpose([lat_i, psi_i])
            new_x_i_t = self.lateral_lpv.change_coordinates(x_i_local_unrotated, back=False, interval=True)
            delta = new_x_i_t.mean(axis=0) - self.lateral_lpv.x_i_t.mean(axis=0)
            self.lateral_lpv.x_i_t += delta
            x_i_local_unrotated = self.longitudinal_lpv.change_coordinates(self.longitudinal_lpv.x_i_t,
                                                                         back=True,
                                                                         interval=True)
            x_i_local_unrotated[:, 0] = longi_i
            new_x_i_t = self.longitudinal_lpv.change_coordinates(x_i_local_unrotated,
                                                                 back=False,
                                                                 interval=True)
            self.longitudinal_lpv.x_i_t += new_x_i_t.mean(axis=0) - self.longitudinal_lpv.x_i_t.mean(axis=0)
            self.previous_target_lane_index = self.target_lane_index

        # Step
        self.longitudinal_lpv.step(dt)
        self.lateral_lpv.step(dt)

        # Backward coordinates change
        x_i_long = self.longitudinal_lpv.change_coordinates(self.longitudinal_lpv.x_i_t, back=True, interval=True)
        x_i_lat = self.lateral_lpv.change_coordinates(self.lateral_lpv.x_i_t, back=True, interval=True)

        # Conversion from rectified to true coordinates
        target_lane = self.road.network.get_lane(self.target_lane_index)
        position_i = interval_local_to_absolute(x_i_long[:, 0], x_i_lat[:, 0], target_lane)
        self.interval.position = position_i
        self.interval.speed = x_i_long[:, 2]
        self.interval.heading = x_i_lat[:, 1]

    def predictor_init(self) -> None:
        """Initialize the LPV models used for interval prediction."""
        position_i = self.interval.position
        target_lane = self.road.network.get_lane(self.target_lane_index)
        longi_i, lat_i = interval_absolute_to_local(position_i, target_lane)
        v_i = self.interval.speed
        psi_i = self.interval.heading - self.lane.heading_at(longi_i.mean())

        # Longitudinal predictor
        if not self.longitudinal_lpv:
            front_interval = self.get_front_interval()

            # LPV specification
            if front_interval:
                f_longi_i, _ = interval_absolute_to_local(front_interval.position, target_lane)
                f_pos = f_longi_i[0]
                f_vel = front_interval.speed[0]
            else:
                f_pos, f_vel = 0, 0
            x0 = [longi_i[0], f_pos, v_i[0], f_vel]
            center = [-self.DISTANCE_WANTED - self.target_speed * self.TIME_WANTED,
                      0,
                      self.target_speed,
                      self.target_speed]
            noise = 1
            b = np.eye(4)
            d = np.array([[1], [0], [0], [0]])
            omega_i = np.array([[-1], [1]]) * noise
            u = [[self.target_speed], [self.target_speed], [0], [0]]
            a0, da = self.longitudinal_matrix_polytope()
            self.longitudinal_lpv = LPV(x0, a0, da, b, d, omega_i, u, center=center)

            # Lateral predictor
            if not self.lateral_lpv:
                # LPV specification
                x0 = [lat_i[0], psi_i[0]]
                center = [0, 0]
                noise = 0.5
                b = np.identity(2)
                d = np.array([[1], [0]])
                omega_i = np.array([[-1], [1]]) * noise
                u = [[0], [0]]
                a0, da = self.lateral_matrix_polytope()
                self.lateral_lpv = LPV(x0, a0, da, b, d, omega_i, u, center=center)

    def longitudinal_matrix_polytope(self) -> Polytope:
        return IntervalVehicle.parameter_box_to_polytope(self.theta_a_i, self.longitudinal_structure)

    def lateral_matrix_polytope(self) -> Polytope:
        return IntervalVehicle.parameter_box_to_polytope(self.theta_b_i, self.lateral_structure)

    @staticmethod
    def parameter_box_to_polytope(parameter_box: np.ndarray, structure: Callable) -> Polytope:
        a, phi = structure()
        a_theta = lambda params: a + np.tensordot(phi, params, axes=[0, 0])
        return polytope(a_theta, parameter_box)

    def get_front_interval(self) -> "VehicleInterval":
        # TODO: For now, we assume the front vehicle follows the models' front vehicle
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if front_vehicle:
            if isinstance(front_vehicle, IntervalVehicle):
                # Use interval from the observer estimate of the front vehicle
                front_interval = front_vehicle.interval
            else:
                # The front vehicle trajectory interval is not being estimated, so it should be considered as certain.
                # We use a new observer created from that current vehicle state, which will have full certainty.
                front_interval = IntervalVehicle.create_from(front_vehicle).interval
        else:
            front_interval = None
        return front_interval

    def get_followed_lanes(self, lane_change_model: str = "model", squeeze: bool = True) -> List[LaneIndex]:
        """
        Get the list of lanes that could be followed by this vehicle.

        :param lane_change_model: - model: assume that the vehicle will follow the lane of its model behaviour.
                                  - all: assume that any lane change decision is possible at any timestep
                                  - right: assume that a right lane change decision is possible at any timestep
        :param squeeze: if True, remove duplicate lanes (at boundaries of the road)
        :return: the list of followed lane indexes
        """
        lanes = []
        if lane_change_model == "model":
            lanes = [self.target_lane_index]
        elif lane_change_model == "all":
            lanes = self.road.network.side_lanes(self.target_lane_index) + [self.target_lane_index]
        elif lane_change_model == "right":
            lanes = [self.target_lane_index]
            _from, _to, _id = self.target_lane_index
            if _id < len(self.road.network.graph[_from][_to]) - 1 \
                    and self.road.network.get_lane((_from, _to, _id + 1)).is_reachable_from(self.position):
                lanes += [(_from, _to, _id + 1)]
            elif not squeeze:
                lanes += [self.target_lane_index]  # Right lane is also current lane
        return lanes

    def partial_observer_step(self, dt: float, alpha: float = 0) -> None:
        """
        Step the boundary parts of the current state interval

        1. Split x_i(t) into two upper and lower intervals x_i_-(t) and x_i_+(t)
        2. Propagate their observer dynamics x_i_-(t+dt) and x_i_+(t+dt)
        3. Merge the resulting intervals together to x_i(t+dt).

        :param dt: timestep [s]
        :param alpha: ratio of the full interval that defines the boundaries
        """
        # 1. Split x_i(t) into two upper and lower intervals x_i_-(t) and x_i_+(t)
        o = self.interval
        v_minus = IntervalVehicle.create_from(self)
        v_minus.interval = copy.deepcopy(self.interval)
        v_minus.interval.position[1, :] = (1 - alpha) * o.position[0, :] + alpha * o.position[1, :]
        v_minus.interval.speed[1] = (1 - alpha) * o.speed[0] + alpha * o.speed[1]
        v_minus.interval.heading[1] = (1 - alpha) * o.heading[0] + alpha * o.heading[1]
        v_plus = IntervalVehicle.create_from(self)
        v_plus.interval = copy.deepcopy(self.interval)
        v_plus.interval.position[0, :] = alpha * o.position[0, :] + (1 - alpha) * o.position[1, :]
        v_plus.interval.speed[0] = alpha * o.speed[0] + (1 - alpha) * o.speed[1]
        v_plus.interval.heading[0] = alpha * o.heading[0] + (1 - alpha) * o.heading[1]
        # 2. Propagate their observer dynamics x_i_-(t+dt) and x_i_+(t+dt)
        v_minus.road = copy.copy(v_minus.road)
        v_minus.road.vehicles = [v if v is not self else v_minus for v in v_minus.road.vehicles]
        v_plus.road = copy.copy(v_plus.road)
        v_plus.road.vehicles = [v if v is not self else v_plus for v in v_plus.road.vehicles]
        v_minus.observer_step(dt)
        v_plus.observer_step(dt)
        # 3. Merge the resulting intervals together to x_i(t+dt).
        self.interval.position = np.array([v_minus.interval.position[0], v_plus.interval.position[1]])
        self.interval.speed = np.array([v_minus.interval.speed[0], v_plus.interval.speed[1]])
        self.interval.heading = np.array([min(v_minus.interval.heading[0], v_plus.interval.heading[0]),
                                          max(v_minus.interval.heading[1], v_plus.interval.heading[1])])

    def store_trajectories(self) -> None:
        """Store the current model, min and max states to a trajectory list."""
        self.trajectory.append(LinearVehicle.create_from(self))
        self.interval_trajectory.append(copy.deepcopy(self.interval))

    def handle_collisions(self, other: 'RoadObject', dt: float) -> None:
        """
        Worst-case collision check.

        For robust planning, we assume that MDPVehicles collide with the uncertainty set of an IntervalVehicle,
        which corresponds to worst-case outcome.

        :param other: the other vehicle
        :param dt: a timestep
        """
        if not isinstance(other, MDPVehicle):
            super().handle_collisions(other)
            return

        if not self.collidable or self.crashed or other is self:
            return

        # Fast rectangular pre-check
        if not utils.point_in_rectangle(other.position,
                                        self.interval.position[0] - self.LENGTH,
                                        self.interval.position[1] + self.LENGTH):
            return

        # Projection of other vehicle to uncertainty rectangle. This is the possible position of this vehicle which is
        # the most likely to collide with other vehicle
        projection = np.minimum(np.maximum(other.position, self.interval.position[0]),
                                self.interval.position[1])
        # Accurate rectangular check
        if utils.rotated_rectangles_intersect((projection, self.LENGTH, self.WIDTH, self.heading),
                                              (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading)):
            self.speed = other.speed = min(self.speed, other.speed)
            self.crashed = other.crashed = True


class VehicleInterval(object):
    def __init__(self, vehicle: Vehicle) -> None:
        self.position = np.array([vehicle.position, vehicle.position], dtype=float)
        self.speed = np.array([vehicle.speed, vehicle.speed], dtype=float)
        self.heading = np.array([vehicle.heading, vehicle.heading], dtype=float)
