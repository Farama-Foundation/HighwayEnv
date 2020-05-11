from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.uncertainty.prediction import IntervalVehicle

FPS = 15


def test_partial():
    road = Road(RoadNetwork.straight_road_network())
    v = IntervalVehicle(road, position=[0, 0], speed=20, heading=0)
    for _ in range(2 * FPS):
        v.step(dt=1/FPS, mode="partial")
        assert v.interval.position[0, 0] <= v.position[0] <= v.interval.position[1, 0]
        assert v.interval.position[0, 1] <= v.position[1] <= v.interval.position[1, 1]
        assert v.interval.heading[0] <= v.heading <= v.interval.heading[1]


def test_predictor():
    road = Road(RoadNetwork.straight_road_network())
    v = IntervalVehicle(road, position=[0, 0], speed=20, heading=0)
    for _ in range(2 * FPS):
        v.step(dt=1/FPS, mode="predictor")
        assert v.interval.position[0, 0] <= v.position[0] <= v.interval.position[1, 0]
        assert v.interval.position[0, 1] <= v.position[1] <= v.interval.position[1, 1]
        assert v.interval.heading[0] <= v.heading <= v.interval.heading[1]
