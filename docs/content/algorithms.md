(algorithms)=

# Agent Examples

Agents solving the HighwayEnv environments are available in the [eleurent/rl-agents](https://github.com/eleurent/rl-agents) and [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) repositories.

See the [quick start guide](https://highway-env.farama.org/quickstart/#quickstart) for training examples and notebooks.

## Deep Q-Network

```{figure} https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/dqn.gif
:alt: DQN agent solving highway-v0
The DQN agent solving highway-v0.
```

This model-free value-based reinforcement learning agent performs Q-learning with function approximation, using a neural network to represent the state-action value function Q.

**Reference implementation:** [eleurent/rl-agents — Deep Q-Network](https://github.com/eleurent/rl-agents/tree/master/rl_agents/agents/deep_q_network)

## Deep Deterministic Policy Gradient

```{figure} https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/ddpg.gif
:alt: DDPG agent solving parking-v0
The DDPG agent solving parking-v0.
```

This model-free policy-based reinforcement learning agent is optimized directly by gradient ascent. It uses Hindsight Experience Replay to efficiently learn how to solve a goal-conditioned task.

**Reference implementation:** [openai/baselines — HER](https://github.com/openai/baselines/tree/master/baselines/her)

## Value Iteration

```{figure} https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/ttcvi.gif
:alt: Value Iteration agent solving highway-v0
The Value Iteration agent solving highway-v0.
```

Value Iteration is only compatible with finite discrete MDPs, so the environment is first approximated by a [finite-mdp environment](https://github.com/eleurent/finite-mdp) using `env.to_finite_mdp()`. This simplified state representation describes the nearby traffic in terms of predicted Time-To-Collision (TTC) on each lane of the road. The transition model is simplistic and assumes that each vehicle will keep driving at a constant speed without changing lanes. This model bias can be a source of mistakes.

The agent then performs a Value Iteration to compute the corresponding optimal state-value function.

**Reference implementation:** [eleurent/rl-agents — Value Iteration](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/dynamic_programming/value_iteration.py)

## Monte-Carlo Tree Search

```{figure} https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/mcts.gif
:alt: MCTS agent solving highway-v0
The MCTS agent solving highway-v0.
```

This agent leverages a transition and reward model to perform a stochastic tree search [(Coulom, 2006)](https://hal.inria.fr/inria-00116992/document) of the optimal trajectory. No particular assumption is required on the state representation or transition model.

**Reference implementation:** [eleurent/rl-agents — Monte-Carlo Tree Search](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/tree_search/mcts.py)
