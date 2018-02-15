from __future__ import division, print_function
import numpy as np
import copy

class Node(object):
    K = 1.0

    def __init__(self, parent, prior=1):
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.count = 0
        self.value = 0

    def select_action(self, temperature):
        return max(self.children.keys(), key=(lambda key: self.children[key].selection_strategy(temperature)))

    def expand(self, actions_distribution):
        actions, probabilities = actions_distribution
        for i in range(len(actions)):
            if actions[i] not in self.children:
                self.children[actions[i]] = Node(self, probabilities[i])

    def update(self, value):
        self.count += 1
        self.value += self.K / self.count * (value - self.value)

    def update_branch(self, value):
        self.update(value)
        if self.parent:
            self.parent.update_branch(value)

    def selection_strategy(self, temperature):
        ucb = temperature * self.prior * np.sqrt(self.parent.count) / (1 + self.count)
        return self.value + ucb

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.value) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


class MCTS(object):
    def __init__(self, prior_policy, rollout_policy, iterations, temperature=1, max_depth=7):

        self.root = Node(parent=None)
        self.prior_policy = prior_policy
        self.rollout_policy = rollout_policy
        self.iterations = iterations
        self.temperature = temperature
        self.max_depth = max_depth

    def run(self, state):
        node = self.root
        value = 0
        depth = self.max_depth
        while depth > 0 and node.children and not state.is_terminal():
            action = node.select_action(self.temperature)
            reward = state.step(action)
            value += reward
            node = node.children[action]
            depth = depth - 1

        if not state.is_terminal() or node == self.root:
            node.expand(self.prior_policy(state))

        value = self.evaluate(state, value, limit=depth)
        node.update_branch(value)

    def evaluate(self, state, value=0, limit=10):
        for _ in range(limit):
            if state.is_terminal():
                break
            actions, probabilities = self.rollout_policy(state)
            action = np.random.choice(actions, 1, p=probabilities)[0]
            reward = state.step(action)
            value += reward
        return value

    def pick_action(self, state):
        for i in range(self.iterations):
            print('Run', i, 'tree', self.root.value)
            state_copy = copy.deepcopy(state)
            self.run(state_copy)
        return self.root.select_action(temperature=0)

    def step(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = Node(None)


def test():
    from highway.simulation import Simulation
    from highway.vehicle import MDPVehicle, IDMVehicle, Obstacle
    from highway.road import Road
    from highway.mdp import RoadMDP
    road = Road.create_random_road(lanes_count=3, lane_width=4.0, vehicles_count=50, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=MDPVehicle)  # solver = mcts
    sim.RECORD_VIDEO = True
    # road.vehicles.append(Obstacle(road, [100., 4.]))
    # road.vehicles.append(Obstacle(road, [100., 8.]))
    # road.vehicles.append(Obstacle(road, [100., 12.]))

    random_policy = lambda state: (np.array(list(RoadMDP.ACTIONS.keys())),
                                   np.ones((len(RoadMDP.ACTIONS)))/len(RoadMDP.ACTIONS))
    mcts = MCTS(random_policy, random_policy, iterations=100)

    while not sim.done:
        sim.handle_events()
        sim.road.act()
        # Planning for ego-vehicle
        policy_call = sim.t % (sim.FPS // (sim.REAL_TIME_RATIO * sim.POLICY_FREQUENCY)) == 0
        if policy_call:
            mdp_state = RoadMDP(sim.vehicle)
            print(len(mdp_state.ego_vehicle.road.vehicles), '->', len(mdp_state.simplified().ego_vehicle.road.vehicles))
            action = mcts.pick_action(mdp_state.simplified())
            sim.vehicle.act(RoadMDP.ACTIONS[action])
            mcts.step(action)
            # self.trajectory = self.vehicle.predict_trajectory(actions,
            #                                                   self.smdp.TIME_QUANTIFICATION,
            #                                                   self.TRAJECTORY_TIMESTEP,
            #                                                   self.dt)
        sim.step()
        sim.display()
    sim.quit()


if __name__ == '__main__':
    test()
