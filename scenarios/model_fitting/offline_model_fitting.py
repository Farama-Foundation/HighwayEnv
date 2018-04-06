from __future__ import division, print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from highway_env.road.road import Road
from highway_env.simulation.simulation import Simulation

from sklearn import linear_model

from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from highway_env.vehicle.dynamics import Obstacle


def generate_data(count):
    # Vehicle.COLLISIONS_ENABLED = False
    vehicle_type = LinearVehicle
    road = Road.create_random_road(lanes_count=2, lane_width=4.0, vehicles_count=5, vehicles_type=vehicle_type)
    sim = Simulation(road, ego_vehicle_type=vehicle_type, displayed=True)
    sim.RECORD_VIDEO = False
    road.add_random_vehicles(5, vehicles_type=vehicle_type)
    road.vehicles.append(Obstacle(road, np.array([50., 0])))
    road.vehicles.append(Obstacle(road, np.array([130., 4.])))
    for v in road.vehicles:
        v.target_velocity = LinearVehicle.VELOCITY_WANTED
        #     v.enable_lane_change = False

    for _ in range(count):
        sim.handle_events()
        sim.act()
        sim.road.dump()
        sim.step()
        sim.display()
    sim.quit()
    return [v.get_log() for v in road.vehicles if not isinstance(v, Obstacle)]


def get_features(data):
    v0 = LinearVehicle.VELOCITY_WANTED
    d_safe = LinearVehicle.DISTANCE_WANTED + data['v'] * LinearVehicle.TIME_WANTED + LinearVehicle.LENGTH

    velocity = v0 - data['v']
    front_velocity = np.minimum(data['front_v'] - data['v'], 0)
    front_distance = np.minimum(data['front_distance'] - d_safe, 0)

    return pd.concat([velocity, front_velocity, front_distance], axis=1).fillna(0)


def fit(dump):
    regr = linear_model.LinearRegression()
    data = dump
    X = get_features(data)
    y = data['acceleration']

    data_fit = data[(-0.95*IDMVehicle.ACC_MAX < data['acceleration'])
                    & (data['acceleration'] < 0.95*IDMVehicle.ACC_MAX)].reset_index(drop=True)
    X_fit = get_features(data_fit)
    y_fit = data_fit['acceleration']

    regr.fit(X_fit, y_fit)
    print(regr.coef_)
    y_pred = np.clip(regr.predict(X), -IDMVehicle.ACC_MAX, IDMVehicle.ACC_MAX)
    # display(y, y_pred)
    return y, y_pred


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
    plt.hist(y_pred - y, bins=30, weights=np.zeros_like(y) + 100. / y.size)
    plt.xlabel(r'Acceleration error [$m/s^2$]')
    plt.ylabel(r'Frequency [%]')
    plt.show()


def main():
    print('Generating data...')
    dumps = generate_data(30 * 30)
    print('Generation done.')
    yy = np.array([])
    yyp = np.array([])
    dumps = [pd.concat(dumps)]
    for d in dumps:
        y, y_pred = fit(d)
        if y is not None:
            yy = np.concatenate((yy, y))
            yyp = np.concatenate((yyp, y_pred))
    display(yy, yyp)


if __name__ == '__main__':
    main()
