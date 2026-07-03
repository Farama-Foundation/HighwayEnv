(faq)=

# Frequently Asked Questions

This is a list of Frequently Asked Questions about HighwayEnv.  Feel free to
suggest new entries!

## When I try to make an environment, I get an error `gymnasium.error.NameNotFound: Environment highway doesn't exist.`

This is probably because you have not imported HighwayEnv yet. Importing HighwayEnv would automatically registers the environments.

```
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)  # this is a no-op to satisfy linters & IDE
```

The last line has no effect, it's simply telling your IDE and/or linter that `highway_env` is actually being used!

## I try to train an agent using the Kinematics Observation and an MLP model, but the resulting policy is not optimal. Why?

I also tend to get reasonable but sub-optimal policies using this observation-model pair.
In {cite}`Leurent2019social`, we argued that a possible reason is that the MLP output depends on the order of
vehicles in the observation. Indeed, if the agent revisits a given scene but observes vehicles described in a different
order, it will see it as a novel state and will not be able to reuse past information. Thus, the agent struggles to
make use of its observation.

This can be addressed in two ways:

- Change the *model*, to use a permutation-invariant architecture which will not be sensitive to the vehicles order, such as *e.g.* {cite}`Qi2017pointnet` or {cite}`Leurent2019social`.

This example is implemented [here (DQN)](https://colab.research.google.com/github/Farama-Foundation/HighwayEnv/blob/main/scripts/intersection_social_dqn.ipynb) or [here (SB3's PPO)](https://github.com/Farama-Foundation/HighwayEnv/blob/main/scripts/sb3_highway_ppo_transformer.py).

- Change the *observation*. For example, the {ref}`Grayscale Image <grayscale-image>` does not depend on an ordering. In this case, a CNN model is more suitable than an MLP model.

This example is implemented [here (SB3's DQN)](https://github.com/Farama-Foundation/HighwayEnv/blob/main/scripts/sb3_highway_dqn_cnn.py).

(faq-uv-frozen)=
## How do I set up a development environment with pinned dependency versions?

We use [uv](https://docs.astral.sh/uv/) to manage development dependencies. The repository includes a `uv.lock` lockfile that pins exact dependency versions known to work together.

To install uv (if you don't have it already):

Install with [standalone installer](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

or

```bash
pip install uv
```

Then clone the repository and sync with frozen (lockfile-pinned) versions:

```bash
git clone https://github.com/Farama-Foundation/HighwayEnv
cd HighwayEnv
uv sync --frozen
```

This creates a virtual environment and installs the project with all its dependencies at the exact versions recorded in the lockfile. To also install test or docs dependencies:

```bash
uv sync --frozen --group test
uv sync --frozen --group docs
uv sync --frozen --group dev   # both test and docs
```

Then run commands through the managed environment with `uv run`:

```bash
uv run pytest
```

## My videos are too fast / have a low framerate.

This is because in gymnasium, a single video frame is generated at each call of `env.step(action)`. However, in HighwayEnv, the policy typically runs at a low-level frequency (e.g. 1 Hz) so that a long action (*e.g.* change lane) actually corresponds to several (typically, 15) simulation frames.
In order to also render these intermediate simulation frames, the following should be done:

```python
import gymnasium as gym

# Wrap the env by a RecordVideo wrapper
env = gym.make("highway-v0")
env = RecordVideo(env, video_folder="run",
              episode_trigger=lambda e: True)  # record all episodes

# Provide the video recorder to the wrapped environment
# so it can send it intermediate simulation frames.
env.unwrapped.set_record_video_wrapper(env)

# Record a video as usual
obs, info = env.reset()
done = truncated = False:
while not (done or truncated):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
env.close()
```
