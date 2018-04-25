from __future__ import division, print_function
import numpy as np
import copy
from highway_env.agent.abstract import AbstractAgent
from highway_env.vehicle.dynamics import Obstacle
from gym.utils import seeding


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
        self.count = 0
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
        self.count += 1
        self.value += self.K / self.count * (total_reward - self.value)

    def update_branch(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.update(total_reward)
        if self.parent:
            self.parent.update_branch(total_reward)

    def selection_strategy(self, temperature):
        """
            Select an action according to its value, prior probability and visit count.

        :param temperature: the exploration parameter, positive or zero.
        :return: the selected action with maximum value and exploration bonus.
        """
        if not self.parent:
            return self.value

        # return self.value + temperature * self.prior * np.sqrt(np.log(self.parent.count) / self.count)
        return self.value + temperature*self.prior/(self.count+1)

    def convert_visits_to_prior_in_branch(self, regularization=0.5):
        """
            For any node in the subtree, convert the distribution of all children visit counts to prior
            probabilities, and reset the visit counts.

        :param regularization: in [0, 1], used to add some probability mass to all children.
                               when 0, the prior is a Boltzmann distribution of visit counts
                               when 1, the prior is a uniform distribution
        """
        self.count = 0
        total_count = sum([(child.count+1) for child in self.children.values()])
        for child in self.children.values():
            child.prior = regularization*(child.count+1)/total_count + regularization/len(self.children)
            child.convert_visits_to_prior_in_branch()

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
    def __init__(self, prior_policy, rollout_policy, iterations, temperature, max_depth=7):
        """
            New MCTS instance.

        :param prior_policy: the prior policy used when expanding and selecting nodes
        :param rollout_policy: the rollout policy used to estimate the value of a leaf node
        :param iterations: the number of iterations
        :param temperature: the temperature of exploration
        :param max_depth: the maximum depth of the tree
        """
        self.root = Node(parent=None)
        self.prior_policy = prior_policy
        self.rollout_policy = rollout_policy
        self.iterations = iterations
        self.temperature = temperature
        self.max_depth = max_depth
        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        """
            Seed the rollout policy randomness source
        :param seed: the seed to be used
        :return: the used seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def run(self, state):
        """
            Run an iteration of Monte-Carlo Tree Search, starting from a given state

        :param state: the initial state
        """
        node = self.root
        total_reward = 0
        depth = self.max_depth
        terminal = False
        while depth > 0 and node.children and not terminal:
            action = node.select_action(temperature=self.temperature)
            _, reward, terminal, _ = state.step(action)
            total_reward += reward
            node = node.children[action]
            depth = depth - 1

        if not node.children and \
                (not terminal or node == self.root):
            node.expand(self.prior_policy(state))

        if not terminal:
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
            actions, probabilities = self.rollout_policy(state)
            action = self.np_random.choice(actions, 1, p=probabilities)[0]
            _, reward, terminal, _ = state.step(action)
            total_reward += reward
            if terminal:
                break
        return total_reward

    def plan(self, state):
        """
            Plan an optimal sequence of actions by running several iterations of MCTS.

        :param state: the initial state
        :return: the list of actions
        """
        for i in range(self.iterations):
            if (i+1) % 10 == 0:
                print(i+1, '/', self.iterations)
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
            Update the MCTS tree when the agent performs an action

        :param action: the chosen action from the root node
        """
        self.step_by_subtree(action)

    def step_by_reset(self):
        """
            Reset the MCTS tree to a root node for the new state.
        """
        self.root = Node(None)

    def step_by_subtree(self, action):
        """
            Replace the MCTS tree by its subtree corresponding to the chosen action.

        :param action: a chosen action from the root node
        """
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            # The selected action was never explored, start a new tree.
            self.step_by_reset()

    def step_by_prior(self, action):
        """
            Replace the MCTS tree by its subtree corresponding to the chosen action, but also convert the visit counts
            to prior probabilities and before resetting them.

        :param action: a chosen action from the root node
        """
        self.step_by_subtree(action)
        self.root.convert_visits_to_prior_in_branch()


class MCTSAgent(AbstractAgent):
    """
        An agent that uses Monte Carlo Tree Search to plan a sequence of action in an MDP.
    """
    def __init__(self,
                 prior_policy=None,
                 rollout_policy=None,
                 iterations=75,
                 temperature=10,
                 assume_vehicle_type=None):
        """
            A new MCTS agent.
        :param prior_policy: The prior distribution over actions given a state
        :param rollout_policy: The distribution over actions used when evaluating leaves
        :param iterations: the number of MCTS iterations
        :param temperature: the temperature of exploration
        :param assume_vehicle_type: the model used to predict the vehicles behavior. If None, the true model is used.
        """
        prior_policy = prior_policy or MCTSAgent.fast_policy
        rollout_policy = rollout_policy or MCTSAgent.random_available_policy
        self.mcts = MCTS(prior_policy, rollout_policy, iterations=iterations, temperature=temperature)
        self.assume_vehicle_type = assume_vehicle_type
        self.previous_action = None

    def plan(self, state):
        """
            Plan an optimal sequence of actions.

            Start by updating the previously found tree with the last action performed.

        :param state: the current state
        :return: the list of actions
        """
        self.mcts.step(self.previous_action)

        if self.assume_vehicle_type:
            state = MCTSAgent.change_agent_model(state, self.assume_vehicle_type)
        actions = self.mcts.plan(state)
        self.previous_action = actions[0]
        return actions

    def reset(self):
        self.mcts.step_by_reset()

    def seed(self, seed=None):
        return self.mcts.seed(seed)

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

    @staticmethod
    def change_agent_model(state, agent_type):
        """
            Change the type of all agents on the road
        :param state: The state describing the road and vehicles
        :param agent_type: The new type of behavior for other vehicles
        :return: a new state with modified behavior model for other agents
        """
        state_copy = copy.deepcopy(state)
        vehicles = state_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not state_copy.vehicle and not isinstance(v, Obstacle):
                vehicles[i] = agent_type.create_from(v)
        return state_copy


class RobustMCTSAgent(AbstractAgent):
    def __init__(self,
                 models,
                 prior_policy=None,
                 rollout_policy=None,
                 iterations=75,
                 temperature=10):
        """
            A new MCTS agent with multiple models.
        :param models: a list of possible transition models
        :param prior_policy: The prior distribution over actions given a state
        :param rollout_policy: The distribution over actions used when evaluating leaves
        :param iterations: the number of MCTS iterations
        :param temperature: the temperature of exploration
        """

        self.agents = [MCTSAgent(prior_policy, rollout_policy, iterations, temperature, assume_vehicle_type=model)
                       for model in models]

    def plan(self, state):
        for agent in self.agents:
            agent.plan(state)

        min_action_values = {k: np.inf for k in state.ACTIONS.keys()}
        for agent in self.agents:
            min_action_values = {k: min(v, agent.mcts.root.children[k].value)
                                 for k, v in min_action_values.items()
                                 if k in agent.mcts.root.children}
        action = max(min_action_values.keys(), key=(lambda key: min_action_values[key]))
        for agent in self.agents:
            agent.previous_action = action

        return [action]
