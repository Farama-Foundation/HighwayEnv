from __future__ import division, print_function
import numpy as np
from highway.road import LineType, StraightLane, SineLane, LanesConcatenation, Road
from highway.simulation import Simulation
from highway.vehicle.behavior import IDMVehicle
from highway.vehicle.control import ControlledVehicle
from highway.vehicle.dynamics import Obstacle


def make_road():
    ends = [100, 100, 100]
    l0 = StraightLane(np.array([0, 0]), 0, 4.0, [LineType.CONTINUOUS, LineType.NONE])
    l1 = StraightLane(np.array([0, 4]), 0, 4.0, [LineType.STRIPED, LineType.CONTINUOUS])

    lc0 = StraightLane(np.array([0, 6.5 + 4 + 4]), 0, 4.0, [LineType.CONTINUOUS, LineType.CONTINUOUS], bounds=[-np.inf, ends[0]])
    amplitude = 3.3
    lc1 = SineLane(lc0.position(ends[0], -amplitude), 0, 4.0, amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2,
                   [LineType.STRIPED, LineType.STRIPED], bounds=[0, ends[1]])
    lc2 = StraightLane(lc1.position(ends[1], 0), 0, 4.0, [LineType.NONE, LineType.CONTINUOUS], bounds=[0, ends[2]])
    l2 = LanesConcatenation([lc0, lc1, lc2])
    road = Road([l0, l1, l2])
    road.vehicles.append(Obstacle(road, lc2.position(ends[2], 0)))
    return road, l1, l2


def make_vehicles(road, highway_lane_2, insertion_lane):
    ego_vehicle = ControlledVehicle(road, highway_lane_2.position(0, 0))
    ego_vehicle.target_velocity = 20
    road.vehicles.append(ego_vehicle)
    # road.vehicles.append(IDMVehicle(road, insertion_lane.position(10, 0), velocity=10))
    v = IDMVehicle(road, insertion_lane.position(30, 0), velocity=10)
    v.target_velocity = 20
    road.vehicles.append(v)
    # road.vehicles.append(IDMVehicle(road, insertion_lane.position(50, 0), velocity=10))
    return ego_vehicle


def run():
    road, highway_lane_2, insertion_lane = make_road()
    ego_vehicle = make_vehicles(road, highway_lane_2, insertion_lane)
    sim = Simulation(road)
    sim.vehicle = ego_vehicle
    sim.RECORD_VIDEO = False

    while not sim.done:
        sim.process()
    sim.quit()


if __name__ == '__main__':
    run()
