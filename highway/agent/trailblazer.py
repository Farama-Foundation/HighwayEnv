from __future__ import division, print_function
import numpy as np
import copy


class MaxNode(object):
    def __init__(self, state, gamma, delta, lambd, eta):
        self.state = state
        self.gamma = gamma
        self.delta = delta
        self.lambd = lambd
        self.eta = eta
        self.K = len(state.get_actions())

        self.children = {}
        for action in state.get_actions():
            self.children[action] = AvgNode(state, action, self.gamma, self.delta, self.lambd, self.eta, self.K)

    def run(self, m, epsilon):
        candidates = self.children
        count = 1
        U = np.inf
        while len(candidates) > 1 and U >= (1 - self.eta)*epsilon:
            sqr = (np.log(self.K*count/(self.delta*epsilon)) + self.gamma / (self.eta - self.gamma) + self.lambd + 1) / count
            U = 2/(1-self.gamma)*np.sqrt(sqr)
            mu = [(b, b.run(count, U*self.eta/(1-self.eta))) for b in candidates.values()]
            mu_sup = max(mu, key=lambda c: c[1])[1]
            candidates = [c[0] for c in mu if c[1] + 2*U/(1-self.eta) >= mu_sup - 2*U/(1-self.eta)]
            count += 1

        if len(candidates) > 1:
            return max(mu, key=lambda c: c[1])[1]
        else:
            return candidates[0].run(m, self.eta*epsilon)


class AvgNode(object):
    def __init__(self, state, action, gamma, delta, lambd, eta, K):
        self.state = state
        self.action = action
        self.gamma = gamma
        self.delta = delta
        self.lambd = lambd
        self.eta = eta
        self.K = K

        self.sampled_nodes = []
        self.r = 0

    def run(self, m, epsilon):
        if epsilon >= 1/(1-self.gamma):
            return 0
        if len(self.sampled_nodes) > m:
            active_nodes = self.sampled_nodes[:m]
        else:
            while len(self.sampled_nodes) < m:
                new_state = copy.deepcopy(self.state)
                new_reward = new_state.step(self.action)
                for node in self.sampled_nodes:
                    if node.state == new_state:
                        self.sampled_nodes.append(node)
                    else:
                        self.sampled_nodes.append(
                            MaxNode(new_state, self.gamma, self.delta, self.lambd, self.eta, self.K))

                self.r += new_reward

            active_nodes = self.sampled_nodes
        # At this point, |active_nodes| == m
        uniques = []
        counts = []
        for s in active_nodes:
            try:
                i = uniques.index(s)
                counts[i] += 1
            except ValueError:
                uniques.append(s)
                counts.append(1)

        mu = 0
        for i in range(len(uniques)):
            nu = uniques[i].run(counts[i], epsilon/self.gamma)
            mu += counts[i]/m*nu
        return self.gamma*mu+self.r/len(self.sampled_nodes)


class TrailBlazer(object):
    def __init__(self, state, gamma, delta, epsilon):
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.eta = np.power(self.gamma, 1/max(2, np.log(1/self.epsilon)))
        self.K = len(state.get_actions())
        l1 = 2*np.log(self.epsilon*(1-self.gamma))**2
        l2 = np.log(np.log(self.K)/(1-self.eta))
        l3 = np.log(self.gamma/self.eta)
        print(l1, l2, l3)
        self.lambd = l1 * l2 / l3
        self.m = (np.log(1/self.delta) + self.lambd) / ((1 - self.gamma) ** 2 * self.epsilon ** 2)
        print('gamma {}'.format(gamma))
        print('delta {}'.format(delta))
        print('epsilon {}'.format(epsilon))
        print('self.eta {}'.format(self.eta))
        print('self.K {}'.format(self.K))
        print('self.alpha {}'.format(self.lambd))
        print('self.m {}'.format(self.m))

        self.root = MaxNode(state, gamma, delta, self.lambd, self.eta)

    def run(self):
        return self.root.run(self.m, self.epsilon/2)


class DummyState(object):
    def __init__(self, num):
        self.num = num

    def step(self, action):
        if self.num != 0:
            return 0
        if action == 0:
            self.num = 1
            return 0
        elif action == 1:
            self.num = 2
            return 1
        elif action == 2:
            self.num = 4
            return 0

    @classmethod
    def get_actions(cls):
        return [0, 1, 2]

    def __eq__(self, other):
        return self.num == other

    def __ne__(self, other):
        return self.num != other


def test():
    s0 = DummyState(0)
    tb = TrailBlazer(s0, gamma=0.9, delta=0.9, epsilon=0.1)
    print(tb.run())


if __name__ == '__main__':
    test()
