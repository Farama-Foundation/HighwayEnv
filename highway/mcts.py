from __future__ import division, print_function
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import pygame
import copy

from highway.agent import Agent
from highway.mdp import RoadMDP


class Node(object):
    K = 1.0

    def __init__(self, parent, prior=1):
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.count = 0
        self.value = 0

    def select_action(self, temperature):
        if self.children:
            return max(self.children.keys(), key=(lambda key: self.children[key].selection_strategy(temperature)))
        else:
            return None

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

    def display(self, surface, origin, size, selected=False):
        norm = mpl.colors.Normalize(vmin=-30, vmax=20)
        cmap = cm.jet_r
        color = cmap(norm(self.value), bytes=True)
        if self.value != 0:
            pygame.draw.rect(surface, color, (origin[0], origin[1], size[0], size[1]), 0)
            if selected:
                red = (255, 0, 0)
                pygame.draw.rect(surface, red,(origin[0], origin[1], size[0], size[1]), 1)
        best_action = self.select_action(temperature=0)
        for a in RoadMDP.ACTIONS:
            if a in self.children:
                action_selected = (selected and (a == best_action))
                self.children[a].display(surface,
                                         (origin[0]+size[0], origin[1]+a*size[1]/len(self.children)),
                                         (size[0], size[1]/len(self.children)),
                                         selected=action_selected)

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.value) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


class MCTS(object):
    def __init__(self, prior_policy, rollout_policy, iterations, temperature=10, max_depth=7):

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
            state_copy = copy.deepcopy(state)
            self.run(state_copy)
        return self.root.select_action(temperature=0)

    def step(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = Node(None)

    def display(self, surface):
        black = (0, 0, 0)
        cell_size = (surface.get_width() // self.max_depth, surface.get_height())
        pygame.draw.rect(surface, black, (0, 0, surface.get_width(), surface.get_height()), 0)
        self.root.display(surface, (-cell_size[0], 0), cell_size, selected=True)


class MCTSAgent(Agent):
    def __init__(self, state=None):
        self.mcts = MCTS(MCTSAgent.random_policy, MCTSAgent.random_policy, iterations=10)

    def plan(self, state):
        action = self.mcts.pick_action(state)
        print([c.value for c in self.mcts.root.children.values()])
        print('action = ', action)
        self.mcts.step(action)
        return [state.ACTIONS[action]]

    @staticmethod
    def random_policy(s):
        return np.array(list(s.ACTIONS.keys())), np.ones((len(s.ACTIONS))) / len(s.ACTIONS)

    def display(self, surface):
        self.mcts.display(surface)




def test():
    from highway.simulation import Simulation
    from highway.vehicle import MDPVehicle, IDMVehicle, Obstacle
    from highway.road import Road
    road = Road.create_random_road(lanes_count=3, lane_width=4.0, vehicles_count=50, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=MDPVehicle, agent_type=MCTSAgent)
    sim.RECORD_VIDEO = True
    # road.vehicles.append(Obstacle(road, [100., 4.]))
    # road.vehicles.append(Obstacle(road, [100., 8.]))
    # road.vehicles.append(Obstacle(road, [100., 12.]))


    while not sim.done:
        sim.handle_events()
        sim.act()
        sim.step()
        sim.display()
    sim.quit()


if __name__ == '__main__':
    test()
