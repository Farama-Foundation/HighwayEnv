from __future__ import division, print_function
import numpy as np

EPSILON = 0.01

def do_on_average_every(duration, dt):
    return np.random.randint(int(duration/dt)) == 0

def constrain(x, a, b):
    return np.minimum(np.maximum(x, a), b)

def not_zero(x):
    if abs(x) > EPSILON:
        return x
    elif x > 0:
        return EPSILON
    else:
        return -EPSILON