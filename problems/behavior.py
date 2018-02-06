from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from highway.vehicle import MDPVehicle, IDMVehicle
from highway.road import Road
from highway.simulation import Simulation
from highway.logger import VehicleLogger
import highway.utils

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def generate_data(count):
    road = Road.create_random_road(lanes_count=4, lane_width=4.0, vehicles_count=100, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=IDMVehicle, displayed=False)
    logger = VehicleLogger()

    dump = []
    for _ in range(count):
        sim.act()
        for v in sim.road.vehicles:
            l = logger.dump(v, road=sim.road)
            dump.append(l)
        sim.step()
    sim.quit()
    return dump


def fit(dump):
    get = lambda s: np.array([l[s] for l in dump])
    regr = linear_model.LinearRegression()
    X = np.array([[1,
                   l['v'],
                   l['front_distance'],
                   1 / l['front_distance'],
                   (l['front_v'] - l['v']) / l['front_distance'],
                   l['v'] / l['front_distance'],
                   l['v']**2,
                   1 / l['front_distance']**2,
                   l['front_v'] - l['v'],
                   (l['front_v'] - l['v'])**2,
                   ] for l in dump])
    y = get('acceleration')
    sel = np.where(y > -4.5)
    regr.fit(X[sel[0], :], y[sel])
    print(regr.coef_)
    acc_pred = highway.utils.constrain(regr.predict(X), -IDMVehicle.BRAKE_ACC, IDMVehicle.ACC_MAX)
    plt.plot(get('acceleration'), get('acceleration'), '.',
             get('acceleration'), acc_pred, '.')
    plt.show()


def display(dump):
    get = lambda s: np.array([l[s] for l in dump])
    distance = np.array([l['front_v'] - l['v'] for l in dump])
    plt.plot(distance, get('acceleration'), '.')
    # plt.plot(get('x'), get('front_x'), '.')
    plt.show()


def main():
    dump = generate_data(300)
    fit(dump)


if __name__ == '__main__':
    main()
