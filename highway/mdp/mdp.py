from __future__ import division, print_function
from abc import ABCMeta, abstractmethod


class MDP(object):
    """
        An MDP state, described by its transition and reward functions.
    """
    metaclass__ = ABCMeta

    @abstractmethod
    def step(self, action):
        """
            Given an action a, transition the internal state s into its next state s', and return a reward r(s,a)
        :param action: the action to be performed from the current state
        :return: the reward associated with the (s,a,s') transition
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_actions(cls):
        """
            Get the action space.
        :return: a dictionary of actions, represented by an id and a label.
                 e.g. {0:'WALK', 1:'JUMP'}
        """
        raise NotImplementedError()

    @abstractmethod
    def get_available_actions(self):
        """
            Get the list of actions available in the current state
        :return: the list of available actions
        """
        return list(self.get_actions().keys())

    @abstractmethod
    def is_terminal(self):
        """
        :return: Whether the current state is terminal
        """
        raise NotImplementedError()