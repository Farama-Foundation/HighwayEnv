from __future__ import division, print_function
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import pygame
import copy

from highway.agent.agent import Agent
from highway.mdp.mdp import MDP


class Node(object):
    K = 1.0

    def __init__(self, parent, prior=1):
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.count = 1
        self.value = 0
        self.state = None

    def select_action(self, temperature=None):
        if self.children:
            if temperature == 0:
                return max(self.children.keys(), key=(lambda key: self.children[key].count))
            else:
                return max(self.children.keys(), key=(lambda key: self.children[key].selection_strategy(temperature)))
        else:
            return None

    def expand(self, actions_distribution):
        actions, probabilities = actions_distribution
        for i in range(len(actions)):
            if actions[i] not in self.children:
                self.children[actions[i]] = Node(self, probabilities[i])

    def update(self, value):
        self.value += self.K / self.count * (value - self.value)
        self.count += 1

    def update_branch(self, value):
        self.update(value)
        if self.parent:
            self.parent.update_branch(value)

    def selection_strategy(self, temperature=None):
        if not self.parent:
            return self.value

        if temperature is None:
            temperature = 30*5

        # return self.value + temperature * self.prior * np.sqrt(np.log(self.parent.count) / self.count)
        return self.value + temperature*self.prior/self.count

    def display(self, surface, origin, size, depth=0,
                selected=False,
                display_exploration=False,
                display_count=False,
                display_text=True):
        # Display node value
        cmap = cm.jet_r
        norm = mpl.colors.Normalize(vmin=-15, vmax=30)
        color = cmap(norm(self.value), bytes=True)
        pygame.draw.rect(surface, color, (origin[0], origin[1], size[0], size[1]), 0)

        # Add exploration bonus
        if display_exploration:
            norm = mpl.colors.Normalize(vmin=-15, vmax=30)
            color = cmap(norm(self.selection_strategy()), bytes=True)
            pygame.draw.polygon(surface, color, [(origin[0], origin[1] + size[1]),
                                                 (origin[0] + size[0], origin[1]),
                                                 (origin[0] + size[0], origin[1] + size[1])], 0)

        # Add node count
        if display_count and depth < 3:
            norm = mpl.colors.Normalize(vmin=0, vmax=100)
            color = cmap(norm(self.count), bytes=True)
            pygame.draw.rect(surface, color, (origin[0], origin[1], 5, 5), 0)

        # Add selection display
        if selected:
            red = (255, 0, 0)
            pygame.draw.rect(surface, red, (origin[0], origin[1], size[0], size[1]), 1)

        if display_text and depth < 3:
            font = pygame.font.Font(None, 15)
            text = font.render("{:.1f} / {:.1f} / {}".format(self.value, self.selection_strategy(), self.count),
                               1, (10, 10, 10), (255, 255, 255))
            text_pos = text.get_rect(centerx=origin[0]+20, centery=origin[1]+5)
            surface.blit(text, text_pos)

        # Recursively display children nodes
        best_action = self.select_action(temperature=0)
        for a in RoadMDP.ACTIONS:
            if a in self.children:
                action_selected = (selected and (a == best_action))
                self.children[a].display(surface,
                                         (origin[0]+size[0], origin[1]+a*size[1]/len(RoadMDP.ACTIONS)),
                                         (size[0], size[1]/len(RoadMDP.ACTIONS)),
                                         depth=depth+1, selected=action_selected)

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.value) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


class MCTS(object):
    def __init__(self, prior_policy, rollout_policy, iterations, max_depth=7):

        self.root = Node(parent=None)
        self.prior_policy = prior_policy
        self.rollout_policy = rollout_policy
        self.iterations = iterations
        self.max_depth = max_depth

    def run(self, state):
        node = self.root
        value = 0
        depth = self.max_depth
        while depth > 0 and node.children and not state.is_terminal():
            action = node.select_action()
            reward = state.step(action)
            value += reward
            node = node.children[action]
            depth = depth - 1

        if not node.children and \
                (not state.is_terminal() or node == self.root):
            node.state = copy.deepcopy(state)
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

    def plan(self, state):
        for i in range(self.iterations):
            state_copy = copy.deepcopy(state)
            self.run(state_copy)
        return self.get_plan()

    def get_plan(self):
        actions = []
        node = self.root
        while node.children:
            action = node.select_action(temperature=0)
            actions.append(action)
            node = node.children[action]
        return actions

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
        self.root.display(surface, (0, 0), cell_size, depth=0, selected=True)


class MCTSAgent(Agent):
    def __init__(self, state=None, iterations=50):
        self.mcts = MCTS(MCTSAgent.idle_policy, MCTSAgent.random_available_policy, iterations=iterations)
        self.previous_action = None

    def plan(self, state):
        self.mcts.step(self.previous_action)
        actions = self.mcts.plan(state)
        self.previous_action = actions[0]
        return [state.ACTIONS[a] for a in actions]

    def get_predicted_states(self):
        states = []
        node = self.mcts.root
        while node.children:
            action = node.select_action(temperature=0)
            states.append(node.state)
            node = node.children[action]
        return states

    def display(self, surface):
        self.mcts.display(surface)

    @staticmethod
    def random_policy(s):
        actions = s.get_actions()
        probabilities = np.ones((len(actions))) / len(actions)
        return list(actions.keys()), probabilities

    @staticmethod
    def random_available_policy(s):
        available_actions = s.get_available_actions()
        probabilities = np.ones((len(available_actions))) / len(available_actions)
        return available_actions, probabilities

    @staticmethod
    def preference_policy(s, action_label, ratio=2):
        allowed_actions = s.get_available_actions()
        for i in range(len(allowed_actions)):
            if s.ACTIONS[allowed_actions[i]] == action_label:
                probabilities = np.ones((len(allowed_actions))) / (len(allowed_actions) - 1 + ratio)
                probabilities[i] *= ratio
                return allowed_actions, probabilities
        return MCTSAgent.random_available_policy(s)

    @staticmethod
    def fast_policy(s):
        return MCTSAgent.preference_policy(s, 'FASTER')

    @staticmethod
    def idle_policy(s):
        return MCTSAgent.preference_policy(s, 'IDLE')


def test():
    from highway.simulation import Simulation
    from highway.road.road import Road
    from highway.vehicle.behavior import IDMVehicle
    from highway.vehicle.control import MDPVehicle
    road = Road.create_random_road(lanes_count=4, lane_width=4.0, vehicles_count=40, vehicles_type=IDMVehicle)
    sim = Simulation(road, ego_vehicle_type=MDPVehicle, agent_type=MCTSAgent)

    while not sim.done:
        sim.process()
    sim.quit()


if __name__ == '__main__':
    test()
