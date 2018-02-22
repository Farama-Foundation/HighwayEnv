from __future__ import division, print_function
import numpy as np
from highway.road import Road
from highway.simulation import Simulation
from highway.vehicle.control import ControlledVehicle

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from highway.vehicle.behavior import LinearVehicle


class LinearEstimator(object):
    RHO = 10.0

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.weights = np.zeros(np.shape(np.transpose(self.vehicle_features())))

    def update(self, dt):
        f = self.vehicle_features()

        # self.vehicle.action['acceleration'] = np.transpose(f).dot(np.array([[1.], [2.], [10]]))[0, 0]
        # print(self.vehicle.action['acceleration'])

        a = self.vehicle.action['acceleration']
        # self.weights += self.RHO*f*(a - np.transpose(f).dot(self.weights))*dt
        if -self.vehicle.BRAKE_ACC < a < self.vehicle.ACC_MAX:
            self.weights = np.linalg.inv(np.eye(3) + dt*self.RHO*f.dot(np.transpose(f))).dot(self.weights + dt*self.RHO*f*a)
        print('w = ', np.transpose(self.weights))

    def vehicle_features(self):
        f = np.zeros((3, 1))
        f[0] = self.vehicle.target_velocity - self.vehicle.velocity
        for v in (self.vehicle.road.neighbour_vehicles(self.vehicle)):
            if not v:
                continue
            dp = self.vehicle.lane_distance_to(v)
            dv = v.velocity - self.vehicle.velocity
            f[1] = LinearEstimator.velocity_feature(dp, dv)
            f[2] = LinearEstimator.position_feature(dp, self.vehicle.velocity, v.velocity)
        return f

    @staticmethod
    def velocity_feature(dp, dv):
        return np.minimum(np.sign(dp)*dv, 0)*np.sign(dp)

    @staticmethod
    def position_feature(dp, vi, vj):
        safe_distance = lambda v: LinearVehicle.LENGTH + LinearVehicle.DISTANCE_WANTED + v * LinearVehicle.TIME_WANTED
        if dp > 0:
            return -np.maximum(safe_distance(vi) - np.abs(dp), 0)
        else:
            return np.maximum(safe_distance(vj) - np.abs(dp), 0)


def main():
    road = Road.create_random_road(lanes_count=2, lane_width=4.0, vehicles_count=1, vehicles_type=LinearVehicle)
    v = road.vehicles[0]
    v.enable_lane_change = False
    le = LinearEstimator(v)
    sim = Simulation(road, ego_vehicle_type=ControlledVehicle)

    while not sim.done:
        sim.handle_events()
        sim.act()
        le.update(sim.dt)
        sim.step()
        sim.display()
    sim.quit()


if __name__ == '__main__':
    main()
