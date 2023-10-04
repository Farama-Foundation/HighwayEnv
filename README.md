# highway-env

[![build](https://github.com/eleurent/highway-env/workflows/build/badge.svg)](https://github.com/eleurent/highway-env/actions?query=workflow%3Abuild)
[![Documentation Status](https://github.com/Farama-Foundation/HighwayEnv/actions/workflows/build-docs-dev.yml/badge.svg)](https://farama-foundation.github.io/HighwayEnv/)
[![Downloads](https://img.shields.io/pypi/dm/highway-env)](https://pypi.org/project/highway-env/)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/63847d9328f64fce9c137b03fcafcc27)](https://app.codacy.com/manual/eleurent/highway-env?utm_source=github.com&utm_medium=referral&utm_content=eleurent/highway-env&utm_campaign=Badge_Grade_Dashboard)
[![GitHub contributors](https://img.shields.io/github/contributors/eleurent/highway-env)](https://github.com/eleurent/highway-env/graphs/contributors)


A collection of environments for *autonomous driving* and tactical decision-making tasks, maintained by [Edouard Leurent](https://github.com/eleurent)

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/highway-env.gif?raw=true"><br/>
    <em>An episode of one of the environments available in highway-env.</em>
</p>

## [Try it on Google Colab! ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](scripts)

## The environments

### Highway

```python
env = gymnasium.make("highway-v0")
```

In this task, the ego-vehicle is driving on a multilane highway populated with other vehicles.
The agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles. Driving on the right side of the road is also rewarded.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/highway.gif?raw=true"><br/>
    <em>The highway-v0 environment.</em>
</p>

A faster variant, `highway-fast-v0` is also available, with a degraded simulation accuracy to improve speed for large-scale training.

### Merge

```python
env = gymnasium.make("merge-v0")
```

In this task, the ego-vehicle starts on a main highway but soon approaches a road junction with incoming vehicles on the access ramp. The agent's objective is now to maintain a high speed while making room for the vehicles so that they can safely merge in the traffic.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/merge-env.gif?raw=true"><br/>
    <em>The merge-v0 environment.</em>
</p>

### Roundabout

```python
env = gymnasium.make("roundabout-v0")
```

In this task, the ego-vehicle if approaching a roundabout with flowing traffic. It will follow its planned route automatically, but has to handle lane changes and longitudinal control to pass the roundabout as fast as possible while avoiding collisions.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/roundabout-env.gif?raw=true"><br/>
    <em>The roundabout-v0 environment.</em>
</p>

### Parking

```python
env = gymnasium.make("parking-v0")
```

A goal-conditioned continuous control task in which the ego-vehicle must park in a given space with the appropriate heading.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/parking-env.gif?raw=true"><br/>
    <em>The parking-v0 environment.</em>
</p>

### Intersection

```python
env = gymnasium.make("intersection-v0")
```

An intersection negotiation task with dense traffic.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/intersection-env.gif?raw=true"><br/>
    <em>The intersection-v0 environment.</em>
</p>

### Racetrack

```python
env = gymnasium.make("racetrack-v0")
```

A continuous control task involving lane-keeping and obstacle avoidance.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/racetrack-env.gif?raw=true"><br/>
    <em>The racetrack-v0 environment.</em>
</p>


## Examples of agents

Agents solving the `highway-env` environments are available in the [eleurent/rl-agents](https://github.com/eleurent/rl-agents) and [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) repositories.

See the [documentation](https://farama-foundation.github.io/HighwayEnv/quickstart/#training-an-agent) for some examples and notebooks.

### [Deep Q-Network](https://github.com/eleurent/rl-agents/tree/master/rl_agents/agents/deep_q_network)

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/dqn.gif?raw=true"><br/>
    <em>The DQN agent solving highway-v0.</em>
</p>

This model-free value-based reinforcement learning agent performs Q-learning with function approximation, using a neural network to represent the state-action value function Q.

### [Deep Deterministic Policy Gradient](https://github.com/openai/baselines/tree/master/baselines/her)

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/ddpg.gif?raw=true"><br/>
    <em>The DDPG agent solving parking-v0.</em>
</p>

This model-free policy-based reinforcement learning agent is optimized directly by gradient ascent. It uses Hindsight Experience Replay to efficiently learn how to solve a goal-conditioned task.

### [Value Iteration](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/dynamic_programming/value_iteration.py)

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/ttcvi.gif?raw=true"><br/>
    <em>The Value Iteration agent solving highway-v0.</em>
</p>

The Value Iteration is only compatible with finite discrete MDPs, so the environment is first approximated by a [finite-mdp environment](https://github.com/eleurent/finite-mdp) using `env.to_finite_mdp()`. This simplified state representation describes the nearby traffic in terms of predicted Time-To-Collision (TTC) on each lane of the road. The transition model is simplistic and assumes that each vehicle will keep driving at a constant speed without changing lanes. This model bias can be a source of mistakes.

The agent then performs a Value Iteration to compute the corresponding optimal state-value function.

### [Monte-Carlo Tree Search](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/tree_search/mcts.py)

This agent leverages a transition and reward models to perform a stochastic tree search [(Coulom, 2006)](https://hal.inria.fr/inria-00116992/document) of the optimal trajectory. No particular assumption is required on the state representation or transition model.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/mcts.gif?raw=true"><br/>
    <em>The MCTS agent solving highway-v0.</em>
</p>

## Installation

`pip install highway-env`

## Usage

```python
import gymnasium as gym

env = gym.make('highway-v0', render_mode='human')

obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = ... # Your agent code here
    obs, reward, done, truncated, info = env.step(action)
```

## Documentation

Read the [documentation online](https://farama-foundation.github.io/HighwayEnv/).

## Citing

If you use the project in your work, please consider citing it with:
```bibtex
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}
```

List of publications & preprints using `highway-env` (please open a pull request to add missing entries):
*   [Approximate Robust Control of Uncertain Dynamical Systems](https://arxiv.org/abs/1903.00220) (Dec 2018)
*   [Interval Prediction for Continuous-Time Systems with Parametric Uncertainties](https://arxiv.org/abs/1904.04727) (Apr 2019)
*   [Practical Open-Loop Optimistic Planning](https://arxiv.org/abs/1904.04700) (Apr 2019)
*   [α^α-Rank: Practically Scaling α-Rank through Stochastic Optimisation](https://arxiv.org/abs/1909.11628) (Sep 2019)
*   [Social Attention for Autonomous Decision-Making in Dense Traffic](https://arxiv.org/abs/1911.12250) (Nov 2019)
*   [Budgeted Reinforcement Learning in Continuous State Space](http://papers.nips.cc/paper/9128-budgeted-reinforcement-learning-in-continuous-state-space/) (Dec 2019)
*   [Multi-View Reinforcement Learning](http://papers.nips.cc/paper/8422-multi-view-reinforcement-learning) (Dec 2019)
*   [Reinforcement learning for Dialogue Systems optimization with user adaptation](https://tel.archives-ouvertes.fr/tel-02422691/) (Dec 2019)
*   [Distributional Soft Actor Critic for Risk Sensitive Learning](https://arxiv.org/abs/2004.14547) (Apr 2020)
*   [Bi-Level Actor-Critic for Multi-Agent Coordination](https://ojs.aaai.org/index.php/AAAI/article/view/6226) (Apr 2020)
*   [Task-Agnostic Online Reinforcement Learning with an Infinite Mixture of Gaussian Processes](https://arxiv.org/abs/2006.11441) (Jun 2020)
*   [Beyond Prioritized Replay: Sampling States in Model-Based RL via Simulated Priorities](https://arxiv.org/abs/2007.09569) (Jul 2020)
*   [Robust-Adaptive Interval Predictive Control for Linear Uncertain Systems](https://arxiv.org/abs/2007.10401) (Jul 2020)
*   [SMART: Simultaneous Multi-Agent Recurrent Trajectory Prediction](https://arxiv.org/abs/2007.13078) (Jul 2020)
*   [Delay-Aware Multi-Agent Reinforcement Learning for Cooperative and Competitive Environments](https://arxiv.org/abs/2005.05441) (Aug 2020)
*   [B-GAP: Behavior-Guided Action Prediction for Autonomous Navigation](https://arxiv.org/abs/2011.03748) (Nov 2020)
*   [Model-based Reinforcement Learning from Signal Temporal Logic Specifications](https://arxiv.org/abs/2011.04950) (Nov 2020)
*   [Robust-Adaptive Control of Linear Systems: beyond Quadratic Costs](https://arxiv.org/abs/2002.10816) (Dec 2020)
*   [Assessing and Accelerating Coverage in Deep Reinforcement Learning](https://arxiv.org/abs/2012.00724) (Dec 2020)
*   [Distributionally Consistent Simulation of Naturalistic Driving Environment for Autonomous Vehicle Testing](https://arxiv.org/abs/2101.02828) (Jan 2021)
*   [Interpretable Policy Specification and Synthesis through Natural Language and RL](https://arxiv.org/abs/2101.07140) (Jan 2021)
*   [Deep Reinforcement Learning Techniques in Diversified Domains: A Survey](https://link.springer.com/article/10.1007/s11831-021-09552-3) (Feb 2021)
*   [Corner Case Generation and Analysis for Safety Assessment of Autonomous Vehicles](https://arxiv.org/abs/2102.03483) (Feb 2021)
*   [Intelligent driving intelligence test for autonomous vehicles with naturalistic and adversarial environment](https://www.nature.com/articles/s41467-021-21007-8) (Feb 2021)
*   [Building Safer Autonomous Agents by Leveraging Risky Driving Behavior Knowledge](https://arxiv.org/abs/2103.10245)
*   [Quick Learner Automated Vehicle Adapting its Roadmanship to Varying Traffic Cultures with Meta Reinforcement Learning](https://arxiv.org/abs/2104.08876) (Apr 2021)
*   [Deep Multi-agent Reinforcement Learning for Highway On-Ramp Merging in Mixed Traffic](https://arxiv.org/abs/2105.05701) (May 2021)
*   [Accelerated Policy Evaluation: Learning Adversarial Environments with Adaptive Importance Sampling](https://arxiv.org/abs/2106.10566) (Jun 2021)
*   [Learning Interaction-aware Guidance Policies for Motion Planning in Dense Traffic Scenarios](https://arxiv.org/abs/2107.04538) (Jul 2021)
*   [Automatic Overtaking on Two-way Roads with Vehicle Interactions Based on Proximal Policy Optimization](https://ieeexplore.ieee.org/abstract/document/9575954) (Jul 2021)
*   [Robust Predictable Control](https://arxiv.org/abs/2109.03214) (Sep 2021)
*   [Improving Robustness of Deep Reinforcement Learning Agents: Environment Attack based on the Critic Network](https://ieeexplore.ieee.org/document/9892901) (Jul 2022)
*   [Autonomous Highway Merging in Mixed Traffic Using Reinforcement Learning and Motion Predictive Safety Controller](https://ieeexplore.ieee.org/document/9921741) (Oct 2022)

PhD theses
*   [Reinforcement learning for Dialogue Systems optimization with user adaptation](https://hal.inria.fr/tel-02422691/) (2019)
*   [Safe and Efficient Reinforcement Learning for Behavioural Planning in Autonomous Driving](https://hal.inria.fr/tel-03035705/) (2020)
*   [Many-agent Reinforcement Learning](https://discovery.ucl.ac.uk/id/eprint/10124273/) (2021)

Master theses
*   [Multi-Agent Reinforcement Learning with Application on Traffic Flow Control](https://www.diva-portal.org/smash/get/diva2:1573441/FULLTEXT01.pdf) (Jun 2021)
*   [Deep Reinforcement Learning for Automated Parking](https://repositorio-aberto.up.pt/bitstream/10216/136074/2/494682.pdf) (Aug 2021)
*   [Deep Reinforcement Learning and Imitation Learning for Autonomous Driving in a Minimalist Environment](https://www.academia.edu/107587654/Deep_Reinforcement_Learning_and_Imitation_Learning_for_Autonomous_Driving_in_a_Minimalist_Environment) (Jun 2021)


