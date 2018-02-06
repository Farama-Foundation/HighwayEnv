from __future__ import division, print_function
import numpy as np


def pre(name, prefix):
    return '_'.join([prefix, name]) if prefix else name


class VehicleLogger(object):
    def __init__(self):
        pass

    def dump(self, vehicle, road=None, prefix=None):
        dump = {
            pre('v', prefix): vehicle.velocity if vehicle else 999,
            pre('acceleration', prefix): vehicle.action['acceleration'] if vehicle else 0,
            pre('steering', prefix): vehicle.action['steering'] if vehicle else 0
        }

        if road:
            # Add front vehicle logs
            front_vehicle, _ = road.neighbour_vehicles(vehicle)
            front_dump = self.dump(front_vehicle, road=None, prefix=pre('front', prefix))
            dump.update(front_dump)

            # Add front relative logs
            if front_vehicle:
                d = vehicle.lane_distance_to_vehicle(front_vehicle) - vehicle.LENGTH / 2 - front_vehicle.LENGTH / 2
            dump[pre('front_distance', prefix)] = d if front_vehicle else 999

        return dump


def test():
    from highway.vehicle import Vehicle
    r = None
    v = Vehicle(r, [0, 0], 0, 20)
    logger = VehicleLogger()
    print(logger.dump(v, r, 'prefix'))


if __name__ == '__main__':
    test()
