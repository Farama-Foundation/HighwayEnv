import copy
import itertools

import numpy as np

from highway_env import utils
from highway_env.interval import polytope, vector_interval_section, integrator_interval, \
    interval_negative_part, intervals_diff, intervals_product, LPV, interval_absolute_to_local, \
    interval_local_to_absolute
from highway_env.vehicle.behavior import LinearVehicle
from highway_env.vehicle.control import MDPVehicle
from highway_env.vehicle.uncertainty.prediction import IntervalVehicle


class RegressionVehicle(IntervalVehicle):
    """
        Estimator for the parameter of a LinearVehicle.
    """
    def estimate(self, data, lambda_=1e-5, sigma=0.05):
        phi = np.array(data["features"])
        y = np.array(data["outputs"])
        G_N_lambda = 1/sigma * np.transpose(phi) @ phi + lambda_ * np.identity(phi.shape[-1])
        theta_N_lambda = np.linalg.inv(G_N_lambda) @ np.transpose(phi) @ y / sigma
        return theta_N_lambda, G_N_lambda

    def parameter_polytope(self, data, delta, param_bound, lambda_=1e-5):
        theta_N_lambda, G_N_lambda = self.estimate(data)
        d = G_N_lambda.shape[0]
        beta_n = np.sqrt(2*np.log(np.sqrt(np.linalg.det(G_N_lambda) / lambda_ ** d) / delta)) + \
                 np.sqrt(lambda_*d) * param_bound
        values, P = np.linalg.eig(G_N_lambda)
        M = np.sqrt(beta_n) * np.linalg.inv(P) @ np.diag(np.sqrt(1 / values))
        h = np.array(list(itertools.product([-1, 1], repeat=d)))
        d_theta = [M @ h_k for h_k in h]
        return theta_N_lambda, d_theta, beta_n, M

    def longitudinal_matrix_polytope(self):
        return self.polytope_from_estimation(self.data["longitudinal"], self.theta_a_i, self.longitudinal_structure)

    def lateral_matrix_polytope(self):
        return self.polytope_from_estimation(self.data["lateral"], self.theta_b_i, self.lateral_structure)

    def polytope_from_estimation(self, data, parameter_box, structure):
        if not data:
            return self.parameter_box_to_polytope(parameter_box, structure)
        # Parameters polytope
        theta_N_lambda, d_theta, _, _ = self.parameter_polytope(data, delta=0.1,
                                                                param_bound=np.amax(parameter_box[1]))
        theta_clipped = np.clip(theta_N_lambda, parameter_box[0], parameter_box[1])
        for k in range(len(d_theta)):
            d_theta[k] = np.clip(d_theta[k], parameter_box[0] - theta_clipped, parameter_box[1] - theta_clipped)

        # Structure
        a, phi = structure()
        a0 = a + np.tensordot(theta_clipped, phi, axes=[0, 0])
        da = [np.tensordot(d_theta_k, phi, axes=[0, 0]) for d_theta_k in d_theta]
        return a0, da


class MultipleModelVehicle(RegressionVehicle):

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
            super().__init__(road, position, heading, velocity, target_lane_index, target_velocity, route,
                             enable_lane_change, timer, theta_a_i, theta_b_i, data)
            self.possible_routes_data = {}

        def collect_data(self):
            self.update_possible_routes()
            for route, data in self.possible_routes_data:
                self.add_features(data, )



        def update_possible_routes(self):
            pass