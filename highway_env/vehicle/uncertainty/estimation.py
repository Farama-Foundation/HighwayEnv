import copy

import numpy as np

from highway_env import utils
from highway_env.interval import polytope, vector_interval_section, integrator_interval, \
    interval_negative_part, intervals_diff, intervals_product, LPV, interval_absolute_to_local, \
    interval_local_to_absolute
from highway_env.vehicle.behavior import LinearVehicle
from highway_env.vehicle.control import MDPVehicle


class RegressionVehicle(LinearVehicle):

    LAMBDA = 0.1
    """
        Estimator for the parameter of a LinearVehicle.
    """
    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 enable_lane_change=True,
                 timer=None,
                 theta_a_i=None,
                 theta_b_i=None,
                 data=None):
        """
        :param theta_a_i: The interval of possible acceleration parameters
        :param theta_b_i: The interval of possible steering parameters
        """
        super().__init__(road,
                         position,
                         heading,
                         velocity,
                         target_lane_index,
                         target_velocity,
                         route,
                         enable_lane_change,
                         timer)
        self.theta_a_i = theta_a_i if theta_a_i is not None else LinearVehicle.ACCELERATION_RANGE
        self.theta_b_i = theta_b_i if theta_b_i is not None else LinearVehicle.STEERING_RANGE
        self.data = data
        if data:
            print(self.estimate(self.data["longitudinal"]), self.ACCELERATION_PARAMETERS)
            print(self.estimate(self.data["lateral"]), self.STEERING_PARAMETERS)

    @classmethod
    def create_from(cls, vehicle):
        v = cls(vehicle.road,
                vehicle.position,
                heading=vehicle.heading,
                velocity=vehicle.velocity,
                target_lane_index=getattr(vehicle, 'target_lane_index', None),
                target_velocity=getattr(vehicle, 'target_velocity', None),
                route=getattr(vehicle, 'route', None),
                timer=getattr(vehicle, 'timer', None),
                theta_a_i=getattr(vehicle, 'theta_a_i', None),
                theta_b_i=getattr(vehicle, 'theta_b_i', None),
                data=getattr(vehicle, "data", None))
        return v

    def estimate(self, data):
        phi = np.array(data["features"])
        y = np.array(data["outputs"])
        G_N_lambda = np.transpose(phi) @ phi + self.LAMBDA * np.identity(phi.shape[-1])
        theta_N_lambda = np.linalg.inv(G_N_lambda) @ np.transpose(phi) @ y
        return theta_N_lambda
