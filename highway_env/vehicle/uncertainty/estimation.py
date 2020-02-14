import numpy as np

from highway_env.utils import confidence_polytope, is_consistent_dataset
from highway_env.vehicle.behavior import LinearVehicle
from highway_env.vehicle.uncertainty.prediction import IntervalVehicle


class RegressionVehicle(IntervalVehicle):
    """
        Estimator for the parameter of a LinearVehicle.
    """
    def longitudinal_matrix_polytope(self):
        return self.polytope_from_estimation(self.data["longitudinal"], self.theta_a_i, self.longitudinal_structure)

    def lateral_matrix_polytope(self):
        return self.polytope_from_estimation(self.data["lateral"], self.theta_b_i, self.lateral_structure)

    def polytope_from_estimation(self, data, parameter_box, structure):
        if not data:
            return self.parameter_box_to_polytope(parameter_box, structure)
        theta_n_lambda, d_theta, _, _ = confidence_polytope(data, parameter_box=parameter_box)
        a, phi = structure()
        a0 = a + np.tensordot(theta_n_lambda, phi, axes=[0, 0])
        da = [np.tensordot(d_theta_k, phi, axes=[0, 0]) for d_theta_k in d_theta]
        return a0, da


class MultipleModelVehicle(LinearVehicle):
    def __init__(self, road, position, heading=0, velocity=0, target_lane_index=None, target_velocity=None, route=None,
                 enable_lane_change=True, timer=None, data=None):
        super().__init__(road, position, heading, velocity, target_lane_index, target_velocity, route,
                         enable_lane_change, timer, data)
        if not self.data:
            self.data = []

    def act(self):
        if self.collecting_data:
            self.update_possible_routes()
        super().act()

    def collect_data(self):
        """
            Collect the features for each possible route, and true observed outputs.
        """
        for route, data in self.data:
            self.add_features(data, route[0], output_lane=self.target_lane_index)

    def update_possible_routes(self):
        """
            Update a list of possible routes that this vehicle could be following.
            - Add routes at the next intersection
            - Step the current lane in each route
            - Reject inconsistent routes
        """

        for route in self.get_routes_at_intersection():  # Candidates
            # Unknown lane -> first lane
            for i in range(len(route)):
                route[i] = route[i] if route[i][2] is not None else (route[i][0], route[i][1], 0)
            # Is this route already considered, or a suffix of a route already considered ?
            for known_route, _ in self.data:
                if known_route == route:
                    break
                elif len(known_route) < len(route) and route[:len(known_route)] == known_route:
                    self.data = [(r, d) if r != known_route else (route, d) for r, d in self.data]
                    break
            else:
                self.data.append((route.copy(), {}))  # Add it

        # Step the lane being followed in each possible route
        for route, _ in self.data:
            if self.road.network.get_lane(route[0]).after_end(self.position):
                route.pop(0)

        # Reject inconsistent hypotheses
        for route, data in self.data.copy():
            if data:
                if not is_consistent_dataset(data["lateral"], parameter_box=LinearVehicle.STEERING_RANGE):
                    self.data.remove((route, data))

    def assume_model_is_valid(self, index):
        """
            Get a copy of this vehicle behaving according to one of its possible routes.
        :param index: index of the route to consider
        :return: a copy of the vehicle
        """
        if not self.data:
            return self.create_from(self)
        index = min(index, len(self.data)-1)
        route, data = self.data[index]
        vehicle = RegressionVehicle.create_from(self)
        vehicle.target_lane_index = route[0]
        vehicle.route = route
        vehicle.data = data
        return vehicle
