from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from highway.vehicle import MDPVehicle, IDMVehicle
from highway.road import Road
from highway.simulation import Simulation
from highway.logger import VehicleLogger

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def generate_data(count):
    road = Road.create_random_road(lanes_count=1, lane_width=4.0, vehicles_count=1, vehicles_type=IDMVehicle)
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
                   l['front_x'] - l['x'] - 5,
                   1/(l['front_x'] - l['x'] - 5),
                   l['v'],
                   np.power(l['v'], 4),
                   l['v']/(l['front_x'] - l['x'] - 5),
                   (l['front_v'] - l['v'])*l['v']/(l['front_x'] - l['x'] - 5)
                   ] for l in dump])
    y = get('acceleration')
    sel = np.where(y > -4)
    regr.fit(X[sel[0], :], y[sel])
    # print(regr.coef_)
    acc_pred = regr.predict(X)
    acc_reconst = 3-3/np.power(20, 4)*X[:, 4]
    print(acc_reconst)
    plt.plot(get('acceleration'), get('acceleration'), '.',
             get('acceleration'), acc_pred, '.',
             get('acceleration'), acc_reconst, '.')
    plt.show()


def display(dump):
    get = lambda s: np.array([l[s] for l in dump])
    distance = np.array([l['front_v'] - l['v'] for l in dump])
    plt.plot(distance, get('acceleration'), '.')
    # plt.plot(get('x'), get('front_x'), '.')
    plt.show()


def main():
    dump = generate_data(100)
    fit(dump)


if __name__ == '__main__':
    main()
