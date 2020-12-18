.. _rewards:

Rewards
############

The reward function is defined in the :py:meth:`~highway_env.envs.common.abstract.AbstractEnv._reward` method, overloaded in every environment.

.. note::
    The choice of an appropriate reward function that yields realistic optimal driving behaviour is a challenging problem, that we do not address in this project.
    In particular, we do not wish to specify every single aspect of the expected driving behaviour inside the reward function, such as keeping a safe distance to the front vehicle.
    Instead, we would rather only specify a reward function as simple and straightforward as possible in order to see adequate behaviour emerge from learning.
    In this perspective, keeping a safe distance is optimal not for being directly rewarded but for robustness against the uncertain behaviour of the leading vehicle, which could brake at any time.

Most environments
-----------------

We generally focus on two features: a vehicle should

- progress quickly on the road;
- avoid collisions.

Thus, the reward function is often composed of a velocity term and a collision term:

.. math::
    R(s,a) = a\frac{v - v_\min}{v_\max - v_\min} - b\,\text{collision}

where :math:`v,\,v_\min,\,v_\max` are the current, minimum and maximum speed of the ego-vehicle respectively, and :math:`a,\,b` are two coefficients.


.. note::
    Since the rewards must be bounded, and the optimal policy is invariant by scaling and shifting rewards, we choose to normalize them in the :math:`[0, 1]` range, by convention.
    Normalizing rewards has also been observed to be practically beneficial in deep reinforcement learning :cite:`Mnih2015`.
    Note that we forbid negative rewards, since they may encourage the agent to prefer terminating an episode early (by causing a collision) rather than risking suffering a negative return if no satisfying trajectory can be found.

In some environments, the weight of the collision penalty can be configured through the `collision_penalty` parameter.

Goal environments
-----------------

In the :ref:`Parking <environments_parking>` environment, however, the reward function must also specify the desired goal destination.
Thus, the velocity term is replaced by a weighted p-norm between the agent state and the goal state.


.. math::
    R(s,a) = -\| s - s_g \|_{W,p}^p - b\,\text{collision}

where :math:`s = [x, y, v_x, v_y, \cos\psi, \sin\psi]`, :math:`s_g = [x_g, y_g, 0, 0, \cos\psi_g, \sin\psi_g]`, and
:math:`\|x\|_{W,p} = (\sum_i |W_i x_i|^p)^{1/p}`. We use a p-norm rather than an Euclidean norm in order to have a narrower spike of rewards at the goal.
