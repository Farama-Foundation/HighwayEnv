from __future__ import division, print_function
import numpy as np
import copy
from highway.agent.abstract import AbstractAgent


class Node(object):
    """
        An MCTS tree node, corresponding to a given state.
    """
    K = 1.0
    """ The value function first-order filter gain"""

    def __init__(self, parent, prior=1):
        """
            New node.

        :param parent: its parent node
        :param prior: its prior probability
        """
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.count = 1
        self.value = 0

    def select_action(self, temperature=None):
        """
            Select an action from the node.
            - if exploration is wanted with some temperature, follow the selection strategy.
            - else, select the action with maximum visit count

        :param temperature: the exploration parameter, positive or zero
        :return: the selected action
        """
        if self.children:
            if temperature == 0:
                return max(self.children.keys(), key=(lambda key: self.children[key].count))
            else:
                return max(self.children.keys(), key=(lambda key: self.children[key].selection_strategy(temperature)))
        else:
            return None

    def expand(self, actions_distribution):
        """
            Expand a leaf node by creating a new child for each available action.

        :param actions_distribution: the list of available actions and their prior probabilities
        """
        actions, probabilities = actions_distribution
        for i in range(len(actions)):
            if actions[i] not in self.children:
                self.children[actions[i]] = Node(self, probabilities[i])

    def update(self, total_reward):
        """
            Update the visit count and value of this node, given a sample of total reward.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.value += self.K / self.count * (total_reward - self.value)
        self.count += 1

    def update_branch(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.update(total_reward)
        if self.parent:
            self.parent.update_branch(total_reward)

    def selection_strategy(self, temperature=None):
        """
            Select an action according to its value, prior probability and visit count.

        :param temperature: the exploration parameter, positive or zero.
        :return: the selected action with maximum value and exploration bonus.
        """
        if not self.parent:
            return self.value

        if temperature is None:
            temperature = 50*5

        # return self.value + temperature * self.prior * np.sqrt(np.log(self.parent.count) / self.count)
        return self.value + temperature*self.prior/self.count

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.value) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


class MCTS(object):
    """
       An implementation of Monte-Carlo Tree Search, with Upper Confidence Tree exploration.
    """
    def __init__(self, prior_policy, rollout_policy, iterations, max_depth=7):
        """
            New MCTS instance.

        :param prior_policy: the prior policy used when expanding and selecting nodes
        :param rollout_policy: the rollout policy used to estimate the value of a leaf node
        :param iterations: the number of iterations
        :param max_depth: the maximum depth of the tree
        """
        self.root = Node(parent=None)
        self.prior_policy = prior_policy
        self.rollout_policy = rollout_policy
        self.iterations = iterations
        self.max_depth = max_depth

    def run(self, state):
        """
            Run an iteration of Monte-Carlo Tree Search, starting from a given state

        :param state: the initial state
        """
        node = self.root
        total_reward = 0
        depth = self.max_depth
        while depth > 0 and node.children and not state.is_terminal():
            action = node.select_action()
            reward = state.step(action)
            total_reward += reward
            node = node.children[action]
            depth = depth - 1

        if not node.children and \
                (not state.is_terminal() or node == self.root):
            node.expand(self.prior_policy(state))

        total_reward = self.evaluate(state, total_reward, limit=depth)
        node.update_branch(total_reward)

    def evaluate(self, state, total_reward=0, limit=10):
        """
            Run the rollout policy to yield a sample of the value of being in a given state.

        :param state: the leaf state.
        :param total_reward: the initial total reward accumulated until now
        :param limit: the maximum number of simulation steps
        :return: the total reward of the rollout trajectory
        """
        for _ in range(limit):
            if state.is_terminal():
                break
            actions, probabilities = self.rollout_policy(state)
            action = np.random.choice(actions, 1, p=probabilities)[0]
            reward = state.step(action)
            total_reward += reward
        return total_reward

    def plan(self, state):
        """
            Plan an optimal sequence of actions by running several iterations of MCTS.

        :param state: the initial state
        :return: the list of actions
        """
        for i in range(self.iterations):
            state_copy = copy.deepcopy(state)
            self.run(state_copy)
        return self.get_plan()

    def get_plan(self):
        """
            Get the optimal action sequence of the current tree by recursively selecting the best action within each
            node with no exploration.

        :return: the list of actions
        """
        actions = []
        node = self.root
        while node.children:
            action = node.select_action(temperature=0)
            actions.append(action)
            node = node.children[action]
        return actions

    def step(self, action):
        """
            Replace the MCTS tree by its subtree corresponding to the chosen action.

        :param action: a chosen action from the root node
        :return: the tree corresponding to the next state
        """
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            # The selected action was never explored, start a new tree.
            self.root = Node(None)


class MCTSAgent(AbstractAgent):
    """
        An agent that uses Monte Carlo Tree Search to plan a sequence of action in an MDP.
    """
    def __init__(self, state=None, iterations=100):
        """
            A new MCTS agent.
        :param state: the current MDP state
        :param iterations: the number of MCTS iterations
        """
        self.mcts = MCTS(MCTSAgent.fast_policy, MCTSAgent.random_available_policy, iterations=iterations)
        self.previous_action = None

    def plan(self, state):
        """
            Plan an optimal sequence of actions.

            Start by updating the previously found tree with the last action performed.

        :param state: the current MDP state
        :return: the list of MDP action labels
        """
        self.mcts.step(self.previous_action)
        actions = self.mcts.plan(state)
        self.previous_action = actions[0]
        return [state.ACTIONS[a] for a in actions]

    @staticmethod
    def random_policy(state):
        """
            Choose actions from a uniform distribution.

        :param state: the current MDP state
        :return: a list of action indexes and a list of their respective probabilities
        """
        actions = state.get_actions()
        probabilities = np.ones((len(actions))) / len(actions)
        return list(actions.keys()), probabilities

    @staticmethod
    def random_available_policy(state):
        """
            Choose actions from a uniform distribution over currently available actions only.

        :param state: the current MDP state
        :return: a list of action indexes and a list of their respective probabilities
        """
        available_actions = state.get_available_actions()
        probabilities = np.ones((len(available_actions))) / len(available_actions)
        return available_actions, probabilities

    @staticmethod
    def preference_policy(state, action_label, ratio=2):
        """
            Choose actions with a distribution over currently available actions that favors a preferred action.

            The preferred action probability is higher than others with a given ratio, and the distribution is uniform
            over the non-preferred available actions.
        :param state: the current state
        :param action_label: the label of the preferred action
        :param ratio: the ratio between the preferred action probability and the other available actions probabilities
        :return: a list of action indexes and a list of their respective probabilities
        """
        allowed_actions = state.get_available_actions()
        for i in range(len(allowed_actions)):
            if state.ACTIONS[allowed_actions[i]] == action_label:
                probabilities = np.ones((len(allowed_actions))) / (len(allowed_actions) - 1 + ratio)
                probabilities[i] *= ratio
                return allowed_actions, probabilities
        return MCTSAgent.random_available_policy(state)

    @staticmethod
    def fast_policy(state):
        """
            A policy that favors the FASTER action.

        :param state: the current state
        :return: a list of action indexes and a list of their respective probabilities
        """
        return MCTSAgent.preference_policy(state, 'FASTER')

    @staticmethod
    def idle_policy(state):
        """
            A policy that favors the IDLE action.

        :param state: the current state
        :return: a list of action indexes and a list of their respective probabilities
        """
        return MCTSAgent.preference_policy(state, 'IDLE')
