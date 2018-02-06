from __future__ import division, print_function
import numpy as np


def pre(name, prefix):
    return '_'.join([prefix, name]) if prefix else name


class VehicleLogger(object):
    def __init__(self):
        pass

    def dump(self, vehicle, road=None, prefix=None):
        default = 0
        dump = {pre('x', prefix): vehicle.position[0] if vehicle else default,
                pre('y', prefix): vehicle.position[1] if vehicle else default,
                pre('psi', prefix): vehicle.heading if vehicle else default,
                pre('v', prefix): vehicle.velocity * np.cos(vehicle.heading) if vehicle else default,
                pre('acceleration', prefix): vehicle.action['acceleration'] if vehicle else default,
                pre('steering', prefix): vehicle.action['steering'] if vehicle else default
                }

        if road:
            front_vehicle, _ = road.neighbour_vehicles(vehicle)
            front_dump = self.dump(front_vehicle, road=None, prefix='front')
            dump.update(front_dump)

        return dump


def test():
    from highway.vehicle import Vehicle
    r = None
    v = Vehicle(r, [0, 0], 0, 20)
    logger = VehicleLogger()
    print(logger.dump(v, r, 'prefix'))


if __name__ == '__main__':
    test()