from __future__ import division, print_function
import pandas as pd
import matplotlib.pyplot as plt
from highway.vehicle import IDMVehicle
from highway.road import Road
from highway.simulation import Simulation
import highway.utils

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def generate_data(count):
    road = Road.create_random_road(lanes_count=3, lane_width=4.0, vehicles_count=30, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=IDMVehicle, displayed=False)

    for _ in range(count):
        sim.act()
        sim.road.dump()
        sim.step()
    sim.quit()
    return sim.road.get_log()


def fit(dump):
    regr = linear_model.LinearRegression()
    d = dump[dump['front_distance'].notnull()
             & (-4 < dump['acceleration'])
             & (dump['acceleration'] < 2)]
    X = pd.concat([d['v'],
                   d['front_distance'],
                   1 / d['front_distance'],
                   (d['front_v'] - d['v']) / d['front_distance'],
                   d['v'] / d['front_distance'],
                   d['v']**2,
                   1 / d['front_distance']**2,
                   d['front_v'] - d['v'],
                   (d['front_v'] - d['v'])**2,
                   ], axis=1)
    y = d['acceleration']
    regr.fit(X, y)
    print(regr.coef_)
    acc_pred = highway.utils.constrain(regr.predict(X), -IDMVehicle.BRAKE_ACC, IDMVehicle.ACC_MAX)

    plt.scatter(y, y)
    plt.scatter(y, acc_pred)
    plt.show()


def display(dump):
    dump = dump[dump['front_distance'].notnull()]
    dump.plot.scatter(x='front_distance', y='acceleration')
    plt.show()


def main():
    dump = generate_data(100)
    print('Generation done.')
    # display(dump)
    fit(dump)


if __name__ == '__main__':
    main()
