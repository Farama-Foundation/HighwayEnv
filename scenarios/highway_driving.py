from highway.agent.mcts import MCTSAgent
from highway.road.road import Road
from highway.simulation.graphics import SimulationWindow
from highway.simulation.simulation import Simulation
from highway.vehicle.control import MDPVehicle
from highway.vehicle.behavior import IDMVehicle


def test(agent_type):
    road = Road.create_random_road(lanes_count=4, lane_width=4.0, vehicles_count=20, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=MDPVehicle, agent_type=agent_type)
    window = SimulationWindow(sim)

    while not window.done:
        window.process()
    window.quit()


if __name__ == '__main__':
    # test(TTCVIAgent)
    for _ in range(10):
        test(MCTSAgent)
