# Try it on Google Colab

This page contains example notebooks to train RL agents on `highway-env` using several RL libraries.

## Using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

> :warning: **Stable Baselines3 does not currently support `gymnasium`**. As of now, these notebooks are only compatible with previous versions of highway-env, such as `highway-env==1.5`

### Highway with DQN [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/sb3_highway_dqn.ipynb)

Train a `highway-v0` policy with DQN.

### Highway with PPO [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/sb3_highway_ppo.ipynb)

Train a `highway-v0` policy with PPO.

### Highway + DQN using a CNN and image observations [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/sb3_highway_dqn_cnn.ipynb)

Train a `highway-v0` policy with DQN, while using image observations and a CNN model architecture.

### Parking with Hindsight Experience Replay [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/parking_her.ipynb)

Train a goal-conditioned `parking-v0` policy using the [HER](https://arxiv.org/abs/1707.01495) implementation from [stable-baselines](https://github.com/DLR-RM/stable-baselines3) and hyperparameters from the [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

## Using [eleurent/rl-agents](https://github.com/eleurent/rl-agents)

### Trajectory Planning on Highway [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/highway_planning.ipynb)

Plan a trajectory on `highway-v0` using the [OPD](https://hal.archives-ouvertes.fr/hal-00830182/) implementation from [rl-agents](https://github.com/eleurent/rl-agents).

### A Model-based Reinforcement Learning tutorial on Parking  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/parking_model_based.ipynb)

A tutorial written for [RLSS 2019](https://rlss.inria.fr/) and demonstrating the principle of model-based reinforcement learning on the `parking-v0` task.

### Intersection with DQN and social attention [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/intersection_social_dqn.ipynb)

Train an `intersection-v0` crossing policy using the [social attention](https://arxiv.org/abs/1911.12250) architecture and the DQN implementation from [rl-agents](https://github.com/eleurent/rl-agents).
