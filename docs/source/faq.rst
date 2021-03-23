.. _faq:

=============================
Frequently Asked Questions
=============================


This is a list of Frequently Asked Questions about highway-env.  Feel free to
suggest new entries!

I try to train an agent using the Kinematics Observation and an MLP model, but the agent does not learn anything useful. Why?
    I have not managed to make it work either, using this observation-model pair.
    In :cite:`Leurent2019social`, we argued that a possible reason is that the MLP output depends on the order of
    vehicles in the observation. Indeed, if the agent revisits a given scene but observes vehicles described in a different
    order, it will see it as a novel state and will not be able to reuse past information. Thus, the agent struggles to
    make use of its observation.

    This can be addressed in two ways:

    * Change the *model*, to use a permutation-invariant architecture which will not be sensitive to the vehicles order, such as *e.g.* :cite:`Qi2017pointnet` or :cite:`Leurent2019social`.
    This example is implemented `here (DQN) <https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/intersection_social_dqn.ipynb>`_ or `here (SB3's PPO) <https://github.com/eleurent/highway-env/blob/master/scripts/stablebaselines_highway_attention_ppo.py>`_.

    * Change the *observation*. For example, the :ref:`Grayscale Image` does not depend on an ordering. In this case, a CNN model is more suitable than an MLP model.
    This example is implemented `here (SB3's DQN) <https://github.com/eleurent/highway-env/blob/master/scripts/stablebaselines_highway_cnn.py>`_.