.. _quickstart:

===============
Getting Started
===============

Making an environment
~~~~~~~~~~~~~~~~~~~~~~

Here is a quick example of how to create an environment, and run an episode with an `IDLE` policy :

.. code-block:: python

  import gym
  import highway_env

  env = gym.make('highway-v0')
  obs = env.reset()
  done = False
  while not done:
      action = env.ACTIONS["IDLE"]
      obs, reward, done, info = env.step(action)
      env.render()

All the environments
~~~~~~~~~~~~~~~~~~~~
Here is a list of all the environments available and their descriptions:

.. toctree::
  :maxdepth: 1

  environments/highway
  environments/merge
  environments/roundabout
  environments/parking
  environments/intersection

Training an agent
~~~~~~~~~~~~~~~~~~

To train Reinforcement Learing agents, libraries such as `rl-agents <https://github.com/eleurent/rl-agents>`_,
`baselines <https://github.com/openai/baselines>`_ or `stable-baselines <https://github.com/hill-a/stable-baselines>`_
can be used:

.. figure:: https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/parking-env.gif

   The highway-parking-v0 environment trained with HER.

.. code-block:: python

  import gym
  import highway_env
  import numpy as np

  from stable_baselines import HER, SAC, DDPG, TD3
  from stable_baselines.ddpg import NormalActionNoise

  env = gym.make("parking-v0")

  # Create 4 artificial transitions per real transition
  n_sampled_goal = 4

  # SAC hyperparams:
  model = HER('MlpPolicy', env, SAC, n_sampled_goal=n_sampled_goal,
              goal_selection_strategy='future',
              verbose=1, buffer_size=int(1e6),
              learning_rate=1e-3,
              gamma=0.95, batch_size=256,
              policy_kwargs=dict(layers=[256, 256, 256]))

  model.learn(int(2e5))
  model.save('her_sac_highway')

  # Load saved model
  model = HER.load('her_sac_highway', env=env)

  obs = env.reset()

  # Evaluate the agent
  episode_reward = 0
  for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    episode_reward += reward
    if done or info.get('is_success', False):
      print("Reward:", episode_reward, "Success?", info.get('is_success', False))
      episode_reward = 0.0
      obs = env.reset()


Try it on Google Colab!
~~~~~~~~~~~~~~~~~~~~~~~

Use these notebooks to train driving policies on `highway-env`.

.. |parking_mb|  image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/parking_model_based.ipynb
.. |planning_hw|  image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/highway_planning.ipynb
.. |parking_her|  image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/parking_her.ipynb

- A Model-based Reinforcement Learning tutorial on Parking |parking_mb|

  A tutorial written for `RLSS 2019 <https://rlss.inria.fr/>`_ and demonstrating the principle of model-based
  reinforcement learning on the `parking-v0` task.

- Trajectory Planning on Highway |planning_hw|

  Plan a trajectory on `highway-v0` using the `OPD` :cite:`Hren2008` implementation from
  `rl-agents <https://github.com/eleurent/rl-agents>`_.

- Parking with Hindsight Experience Replay |parking_her|

  Train a goal-conditioned `parking-v0` policy using the :cite:`Andrychowicz2017` implementation
  from `stable-baselines <https://github.com/hill-a/stable-baselines>`_.
