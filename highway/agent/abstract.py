from __future__ import division, print_function
from abc import ABCMeta, abstractmethod


class Agent(object):
    metaclass__ = ABCMeta

    @abstractmethod
    def plan(self, state):
        """
            Plan an optimal trajectory given an initial MDP state.

        :param state: the initial state
        :return: the optimal sequence of actions [a0, a1, a2...]
        """
        raise NotImplementedError()
