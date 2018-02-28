from highway.road.road import Road
from highway.simulation import Simulation
from highway.vehicle.control import MDPVehicle


def test():
    from highway.vehicle.behavior import IDMVehicle
    road = Road.create_random_road(lanes_count=4, lane_width=4.0, vehicles_count=5, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=MDPVehicle)
    while not sim.done:
        sim.process()
    sim.quit()


if __name__ == '__main__':
    test()
