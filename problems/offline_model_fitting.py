from __future__ import division, print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from highway.vehicle import IDMVehicle, LinearVehicle
from highway.road import Road
from highway.simulation import Simulation
import highway.utils

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def generate_data(count):
    road = Road.create_random_road(lanes_count=2, lane_width=4.0, vehicles_count=50, vehicles_type=LinearVehicle)
    sim = Simulation(road, ego_vehicle_type=LinearVehicle, displayed=False)

    for _ in range(count):
        sim.act()
        sim.road.dump()
        sim.step()
    sim.quit()
    return sim.road.get_log()


def fit(dump):
    regr = linear_model.LinearRegression()
    d = dump[dump['front_distance'].notnull()
             & dump['rear_distance'].notnull()
             & (-4 < dump['acceleration'])
             & (dump['acceleration'] < 2)].reset_index(drop=True)
    d_safe = LinearVehicle.DISTANCE_WANTED + d['v'] * LinearVehicle.TIME_WANTED + LinearVehicle.LENGTH
    X = pd.concat([d['v'],
                   np.minimum(d['front_v'] - d['v'], 0) + np.maximum(d['rear_v'] - d['v'], 0),
                   np.minimum(d['front_distance'] - d_safe, 0),
                   np.maximum(d_safe - d['rear_distance'], 0)
                   ], axis=1)
    y = d['acceleration']
    regr.fit(X, y)
    print(regr.coef_)
    acc_pred = np.clip(regr.predict(X), -IDMVehicle.BRAKE_ACC, IDMVehicle.ACC_MAX)

    plt.scatter(y, y)
    plt.scatter(y, acc_pred)
    plt.show()


def display(dump):
    dump = dump[dump['front_distance'].notnull()]
    dump.plot.scatter(x='front_distance', y='acceleration')
    plt.show()


def main():
    print('Generating data...')
    dump = generate_data(30*5)
    print('Generation done.')
    # display(dump)
    fit(dump)


if __name__ == '__main__':
    main()
