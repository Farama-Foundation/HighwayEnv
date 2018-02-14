from __future__ import division, print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from highway.vehicle import Vehicle, IDMVehicle, LinearVehicle, Obstacle
from highway.road import Road
from highway.simulation import Simulation
import highway.utils

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def generate_data(count):
    Vehicle.COLLISIONS_ENABLED = False
    road = Road.create_random_road(lanes_count=2, lane_width=4.0, vehicles_count=5, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=IDMVehicle, displayed=True)
    road.vehicles.append(Obstacle(road, road.vehicles[0].position + np.array([30, 0])))

    for _ in range(count):
        sim.handle_events()
        sim.act()
        sim.road.dump()
        sim.step()
        sim.display()
    sim.quit()
    return sim.road.get_log()


def get_features(data):
    v0 = LinearVehicle.VELOCITY_WANTED
    d_safe = LinearVehicle.DISTANCE_WANTED + data['v'] * LinearVehicle.TIME_WANTED + LinearVehicle.LENGTH
    return pd.concat([v0 - data['v'],
                      np.minimum(data['front_v'] - data['v'], 0),
                      np.maximum(data['rear_v'] - data['v'], 0),
                      np.minimum(data['front_distance'] - d_safe, 0),
                      np.maximum(d_safe - data['rear_distance'], 0)
                      ], axis=1)


def fit(dump):
    regr = linear_model.LinearRegression()
    data = dump[dump['front_distance'].notnull()
                & dump['rear_distance'].notnull()].reset_index(drop=True)

    X = get_features(data)
    y = data['acceleration']

    data_fit = data[(-4.9 < data['acceleration'])
                    & (data['acceleration'] < 2.9)].reset_index(drop=True)
    X_fit = get_features(data_fit)
    y_fit = data_fit['acceleration']

    regr.fit(X_fit, y_fit)
    print(regr.coef_)
    y_pred = np.clip(regr.predict(X), -IDMVehicle.BRAKE_ACC, IDMVehicle.ACC_MAX)
    display(y, y_pred)

    regr.coef_[0] = 1
    regr.coef_[1] = 2
    regr.coef_[2] = 0
    regr.coef_[3] = 10
    regr.coef_[4] = 0
    y_pred = np.clip(regr.predict(X), -IDMVehicle.BRAKE_ACC, IDMVehicle.ACC_MAX)
    display(y, y_pred)


def display(y, y_pred):
    plt.figure()
    plt.scatter(y, y, label=r'True')
    plt.scatter(y, y_pred, label=r'Model')
    plt.legend()
    plt.xlabel(r'True acceleration [$m/s^2$]')
    plt.ylabel(r'Acceleration [$m/s^2$]')
    plt.show()

    plt.figure()
    plt.plot(np.arange(np.size(y)), y, label=r'True')
    plt.plot(np.arange(np.size(y)), y_pred, label=r'Model')
    plt.xlabel(r'Time [step]')
    plt.ylabel(r'Acceleration [$m/s^2$]')
    plt.show()

    plt.figure()
    plt.hist(y_pred - y, weights=np.zeros_like(y) + 100. / y.size)
    plt.xlabel(r'Acceleration error [$m/s^2$]')
    plt.ylabel(r'Frequency [%]')
    plt.show()


def main():
    print('Generating data...')
    dump = generate_data(30 * 10)
    print('Generation done.')
    # display(dump)
    fit(dump)


if __name__ == '__main__':
    main()
