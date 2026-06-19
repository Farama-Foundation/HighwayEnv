[![Python](https://img.shields.io/pypi/pyversions/highway-env.svg)](https://badge.fury.io/py/highway-env)
[![PyPI](https://badge.fury.io/py/highway-env.svg)](https://badge.fury.io/py/highway-env)
[![build](https://github.com/Farama-Foundation/HighwayEnv/actions/workflows/build.yml/badge.svg)](https://github.com/Farama-Foundation/HighwayEnv/actions/workflows/build.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
    <a href="https://highway-env.farama.org/" target="_blank">
        <img src="https://github.com/Farama-Foundation/HighwayEnv/blob/main/docs/_static/img/highway-text.png?raw=true" width="500px" />
    </a>
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/HighwayEnv/gh-media/docs/media/highway-env.gif" width="600px" /><br/>
    <em>An episode of the highway-v0 environment.</em>
</p>

HighwayEnv is an open source Python library of environments for autonomous driving and tactical decision-making tasks. The environments include highway, merge, roundabout, parking, intersection, and racetrack scenarios, all following the [Gymnasium](https://gymnasium.farama.org/) API.

The documentation website is at [highway-env.farama.org](https://highway-env.farama.org/), and we have a public discord server (which we also use to coordinate development work) that you can join here: https://discord.gg/bnJ6kubTg6

## Installation

To install HighwayEnv, use:

```bash
pip install highway-env
```

We support Python 3.10+ on Linux and macOS.

## Environments

HighwayEnv includes 6 driving scenario environments: `highway`, `merge`, `roundabout`, `parking`, `intersection`, and `racetrack`. The full list with descriptions and configuration options is available in the [documentation](https://highway-env.farama.org/environments/highway/).

## Usage

```python
import gymnasium as gym

env = gym.make('highway-v0', render_mode='human')

obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = ...  # Your agent code here
    obs, reward, done, truncated, info = env.step(action)
env.close()
```

See the [documentation](https://highway-env.farama.org/quickstart/) for more examples including how to train agents with Stable Baselines3 and Google Colab notebooks. For examples of trained agents (DQN, DDPG, Value Iteration, MCTS), see the [Agent Examples](https://highway-env.farama.org/content/algorithms/) page.

## Development Roadmap

Development on HighwayEnv is ongoing to bring it up to [Farama Standards](https://farama.org/project_standards) for mature projects. See the [GitHub Issues](https://github.com/Farama-Foundation/HighwayEnv/issues) for current open tasks and planned improvements. If you'd like to help out, you can join our Discord server here: https://discord.gg/bnJ6kubTg6

## Project Maintainers

Project Manager: [Edouard Leurent](https://github.com/eleurent)

Maintenance for this project is also contributed by the broader Farama team: [farama.org/team](https://farama.org/team).

## Citation

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

A list of publications using HighwayEnv can be found in the [documentation](https://highway-env.farama.org/content/publications/).
