from highway.agent.mcts import MCTSAgent
from highway.agent.ttc_vi import TTCVIAgent
from highway.road.road import Road
from highway.simulation import Simulation
from highway.vehicle.control import MDPVehicle
from highway.vehicle.behavior import IDMVehicle


def test(agent_type):
    road = Road.create_random_road(lanes_count=4, lane_width=4.0, vehicles_count=30, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=MDPVehicle, agent_type=agent_type)

    while not sim.done:
        sim.process()
    sim.quit()


if __name__ == '__main__':
    test(TTCVIAgent)
    # test(MCTSAgent)
