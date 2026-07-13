[![Python](https://img.shields.io/pypi/pyversions/highway-env.svg)](https://badge.fury.io/py/highway-env)
[![PyPI](https://badge.fury.io/py/highway-env.svg)](https://badge.fury.io/py/highway-env)
[![build](https://github.com/Farama-Foundation/HighwayEnv/actions/workflows/build.yml/badge.svg)](https://github.com/Farama-Foundation/HighwayEnv/actions/workflows/build.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
    <a href="https://highway-env.farama.org/" target="_blank">
        <img src="https://highway-env.farama.org/main/_static/img/highway-text.png" width="500px" />
    </a>
</p>

<p align="center">
    <img src="https://highway-env.farama.org/main/_static/animations/highway-env.gif"><br/>
    <em>An episode of one of the environments available in HighwayEnv.</em>
</p>

A collection of environments for autonomous driving and tactical decision-making tasks. Originally developed by [Edouard Leurent](https://github.com/eleurent) and currently maintained by [Jin Huang](https://github.com/Trenza1ore).

The documentation website is at [highway-env.farama.org](https://highway-env.farama.org), and we have a public discord server (which we also use to coordinate development work) that you can join here: https://discord.gg/bnJ6kubTg6

## Installation

To install HighwayEnv, use:

```bash
pip install highway-env
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add highway-env          # adds to project dependencies and installs (preferred)
uv pip install highway-env  # or install without adding to a project (pip install)
```

We support **Linux** and **macOS** primarily, with **Windows** support maintained on a best-effort basis.

## Environments

HighwayEnv includes 10 driving scenario families: `highway`, `intersection`, `exit`, `lane-keeping`, `merge`, `parking`, `racetrack`, `roundabout`, `two-way`, and `u-turn`, with several environments also offering fast, continuous-control, connected-lane, multi-agent, generic, large, or oval variants. The full list with descriptions and configuration options is available in the [documentation](https://highway-env.farama.org/main/environments/).

<details>
<summary>Previews</summary>

| | |
|:---|:---:|
| `highway` | ![highway](https://highway-env.farama.org/main/_static/animations/environments/highway.gif) |
| `merge` | ![merge](https://highway-env.farama.org/main/_static/animations/environments/merge-env.gif) |
| `roundabout` | ![roundabout](https://highway-env.farama.org/main/_static/animations/environments/roundabout-env.gif) |
| `parking` | ![parking](https://highway-env.farama.org/main/_static/animations/environments/parking-env.gif) |
| `intersection` | ![intersection](https://highway-env.farama.org/main/_static/animations/environments/intersection-env.gif) |
| `racetrack` | ![racetrack](https://highway-env.farama.org/main/_static/animations/environments/racetrack-env.gif) |
| `lane-keeping` | ![lane-keeping](https://highway-env.farama.org/main/_static/animations/environments/lane-keeping-env.gif) |
| `two-way` | ![two-way](https://highway-env.farama.org/main/_static/animations/environments/two-way-env.gif) |
| `exit` | ![exit](https://highway-env.farama.org/main/_static/animations/environments/exit-env.gif) |
| `u-turn` | ![u-turn](https://highway-env.farama.org/main/_static/animations/environments/u-turn-env.gif) |

</details>

## Usage

```python
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)

# Initialise the environment
env = gym.make("highway-v0", config={"lanes_count": 3}, render_mode="human")

# Reset the environment to generate the first observation
obs, info = env.reset()
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    obs, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

See the [documentation](https://highway-env.farama.org/quickstart/) for more examples including how to train agents with Stable Baselines3 and Google Colab notebooks. For examples of trained agents (DQN, DDPG, Value Iteration, MCTS), see the [Agent Examples](https://highway-env.farama.org/content/algorithms/) page.

## Documentation

Read the [documentation online](https://farama-foundation.github.io/HighwayEnv/).

## Development Roadmap

Here is the [roadmap](https://github.com/Farama-Foundation/HighwayEnv/issues/539) for future development work.

## Citating

If you use HighwayEnv in your work, please consider citing it with:

```bibtex
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Farama-Foundation/HighwayEnv}},
}
```

## Publications

A list of publications using HighwayEnv can be found in the [documentation](https://highway-env.farama.org/main/content/publications/).
