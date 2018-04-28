import copy

from highway_env.agent.abstract import AbstractAgent


class SingleTrajectoryAgent(AbstractAgent):
    """
        Execute a given list of actions
    """
    def __init__(self, actions, default_action):
        self.actions = actions
        self.default_action = default_action

    def plan(self, state):
        if self.actions:
            actions = copy.deepcopy(self.actions)
            self.actions.pop(0)
            return actions
        else:
            return [self.default_action]

    def seed(self, seed=None):
        return None

    def reset(self):
        pass


