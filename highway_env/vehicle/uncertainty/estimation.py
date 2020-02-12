import copy
import itertools

import numpy as np

from highway_env.vehicle.behavior import LinearVehicle
from highway_env.vehicle.uncertainty.prediction import IntervalVehicle


class RegressionVehicle(IntervalVehicle):
    """
        Estimator for the parameter of a LinearVehicle.
    """
    @staticmethod
    def estimate(data, lambda_=1e-5, sigma=0.05):
        phi = np.array(data["features"])
        y = np.array(data["outputs"])
        g_n_lambda = 1/sigma * np.transpose(phi) @ phi + lambda_ * np.identity(phi.shape[-1])
        theta_n_lambda = np.linalg.inv(g_n_lambda) @ np.transpose(phi) @ y / sigma
        return theta_n_lambda, g_n_lambda

    @staticmethod
    def parameter_polytope(data, delta, parameter_box, lambda_=1e-5):
        param_bound = np.amax(np.abs(parameter_box))
        theta_n_lambda, g_n_lambda = RegressionVehicle.estimate(data)
        d = g_n_lambda.shape[0]
        beta_n = np.sqrt(2*np.log(np.sqrt(np.linalg.det(g_n_lambda) / lambda_ ** d) / delta)) + \
                 np.sqrt(lambda_*d) * param_bound
        values, pp = np.linalg.eig(g_n_lambda)
        M = np.sqrt(beta_n) * np.linalg.inv(pp) @ np.diag(np.sqrt(1 / values))
        h = np.array(list(itertools.product([-1, 1], repeat=d)))
        d_theta = [M @ h_k for h_k in h]

        # Clip the parameter and confidence region within the prior parameter box.
        theta_n_lambda = np.clip(theta_n_lambda, parameter_box[0], parameter_box[1])
        for k in range(len(d_theta)):
            d_theta[k] = np.clip(d_theta[k], parameter_box[0] - theta_n_lambda, parameter_box[1] - theta_n_lambda)

        return theta_n_lambda, d_theta, beta_n, M

    @staticmethod
    def is_valid_observation(y, phi, theta, beta):
        y_hat = np.tensordot(theta, phi, axes=[0, 0])
        error = np.linalg.norm(y - y_hat)
        eig_values, _ = np.linalg.eig(phi.transpose() @ phi)
        error_bound = np.sqrt(np.amax(eig_values)) * beta
        return error < error_bound

    @staticmethod
    def is_consistent_dataset(data, parameter_box=None):
        train_set = copy.deepcopy(data)
        y, phi = train_set["outputs"].pop(-1), train_set["features"].pop(-1)
        y, phi = np.array(y)[..., np.newaxis], np.array(phi)[..., np.newaxis]
        if train_set["outputs"] and train_set["features"]:
            theta, _, beta, _ = RegressionVehicle.parameter_polytope(train_set, delta=0.1,
                                                                     parameter_box=parameter_box, lambda_=1e-5)
            return RegressionVehicle.is_valid_observation(y, phi, theta, beta)
        else:
            return True

    def longitudinal_matrix_polytope(self):
        data = self.get_data()
        return self.polytope_from_estimation(data["longitudinal"], self.theta_a_i, self.longitudinal_structure)

    def lateral_matrix_polytope(self):
        data = self.get_data()
        return self.polytope_from_estimation(data["lateral"], self.theta_b_i, self.lateral_structure)

    def polytope_from_estimation(self, data, parameter_box, structure):
        if not data:
            return self.parameter_box_to_polytope(parameter_box, structure)
        # Parameters polytope
        theta_n_lambda, d_theta, _, _ = self.parameter_polytope(data, delta=0.1, parameter_box=parameter_box)
        # Structure
        a, phi = structure()
        a0 = a + np.tensordot(theta_n_lambda, phi, axes=[0, 0])
        da = [np.tensordot(d_theta_k, phi, axes=[0, 0]) for d_theta_k in d_theta]
        return a0, da


class MultipleModelVehicle(LinearVehicle):
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
                     data=None):
            super().__init__(road, position, heading, velocity, target_lane_index, target_velocity, route,
                             enable_lane_change, timer, data)
            if not self.data:
                self.data = []

        def act(self):
            self.update_possible_routes()
            super().act()

        def get_data(self):
            for route, data in self.data:
                if route[0] == self.target_lane_index:
                    return data

        def collect_data(self):
            for route, data in self.data:
                self.add_features(data, route[0], output_lane=self.target_lane_index)

        def update_possible_routes(self):
            for route in self.get_routes_at_intersection():
                for i in range(len(route)):
                    route[i] = route[i] if route[i][2] is not None else (route[i][0], route[i][1], 0)
                for known_route, _ in self.data:
                    if known_route == route:
                        break
                    elif len(known_route) < len(route) and route[:len(known_route)] == known_route:
                        self.data = [(r, d) if r != known_route else (route, d) for r, d in self.data]
                        break
                else:
                    self.data.append((route.copy(), {}))

            # Step the lane in each possible route
            for route, _ in self.data:
                if self.road.network.get_lane(route[0]).after_end(self.position):
                    route.pop(0)

            # Reject hypotheses
            for route, data in self.data.copy():
                if data:
                    if not RegressionVehicle.is_consistent_dataset(data["lateral"],
                                                                   parameter_box=LinearVehicle.STEERING_RANGE):
                        self.data.remove((route, data))

        def assume_model_is_valid(self, index):
            if not self.data:
                return self.create_from(self)
            index = min(index, len(self.data)-1)
            route, data = self.data[index]
            vehicle = RegressionVehicle.create_from(self)
            vehicle.target_lane_index = route[0]
            vehicle.route = route
            vehicle.data = data
            return vehicle
