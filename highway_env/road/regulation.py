import numpy as np

from highway_env import utils
from highway_env.road.road import Road
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle


class RegulatedRoad(Road):
    YIELDING_COLOR = None
    REGULATION_FREQUENCY = 2
    YIELD_DURATION = 0.

    def __init__(self, network=None, vehicles=None, np_random=None, record_history=False):
        super().__init__(network, vehicles, np_random, record_history)
        self.steps = 0

    def step(self, dt):
        self.steps += 1
        if self.steps % int(1 / dt / self.REGULATION_FREQUENCY) == 0:
            self.enforce_road_rules()
        return super().step(dt)

    def enforce_road_rules(self):
        """
            Find conflicts and resolve them by assigning yielding vehicles and stopping them.
        """
        # Unfreeze previous yielding vehicles
        for v in self.vehicles:
            if getattr(v, "is_yielding", False):
                if v.yield_timer >= self.YIELD_DURATION * self.REGULATION_FREQUENCY:
                    v.target_velocity = v.lane.speed_limit
                    delattr(v, "color")
                    v.is_yielding = False
                else:
                    v.yield_timer += 1

        # Find new conflicts and resolve them
        for i in range(len(self.vehicles) - 1):
            for j in range(i+1, len(self.vehicles)):
                if self.is_conflict_possible(self.vehicles[i], self.vehicles[j]):
                    yielding_vehicle = self.respect_priorities(self.vehicles[i], self.vehicles[j])
                    if yielding_vehicle is not None and \
                            isinstance(yielding_vehicle, ControlledVehicle) and \
                            not isinstance(yielding_vehicle, MDPVehicle):
                        yielding_vehicle.color = self.YIELDING_COLOR
                        yielding_vehicle.target_velocity = 0
                        yielding_vehicle.is_yielding = True
                        yielding_vehicle.yield_timer = 0

    @staticmethod
    def respect_priorities(v1, v2):
        """
            Resolve a conflict between two vehicles by determining who should yield
        :return: the yielding vehicle
        """
        if v1.lane.priority > v2.lane.priority:
            return v2
        elif v1.lane.priority < v2.lane.priority:
            return v1
        else:  # The vehicle behind should yield
            return v1 if v1.front_distance_to(v2) > v2.front_distance_to(v1) else v2

    @staticmethod
    def is_conflict_possible(v1, v2, horizon=3, step=0.25):
        times = np.arange(step, horizon, step)
        positions_1, headings_1 = v1.predict_trajectory_constant_velocity(times)
        positions_2, headings_2 = v2.predict_trajectory_constant_velocity(times)

        for position_1, heading_1, position_2, heading_2 in zip(positions_1, headings_1, positions_2, headings_2):
            # Fast spherical pre-check
            if np.linalg.norm(position_2 - position_1) > v1.LENGTH:
                continue

            # Accurate rectangular check
            if utils.rotated_rectangles_intersect((position_1, 1.5*v1.LENGTH, 0.9*v1.WIDTH, heading_1),
                                                  (position_2, 1.5*v2.LENGTH, 0.9*v2.WIDTH, heading_2)):
                return True
