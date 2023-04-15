(faq)=

# Frequently Asked Questions

This is a list of Frequently Asked Questions about highway-env.  Feel free to
suggest new entries!

#### When I try to make an environment, I get an error `gymnasium.error.NameNotFound: Environment highway doesn't exist.`

This is probably because you do not have highway-env installed, but are instead working with a local copy of the
repository. In that case, you need to run the following code first to register the environments.

```python
import highway_env
highway_env.register_highway_envs()
```

#### I try to train an agent using the Kinematics Observation and an MLP model, but the resulting policy is not optimal. Why?

I also tend to get reasonable but sub-optimal policies using this observation-model pair.
In {cite}`Leurent2019social`, we argued that a possible reason is that the MLP output depends on the order of
vehicles in the observation. Indeed, if the agent revisits a given scene but observes vehicles described in a different
order, it will see it as a novel state and will not be able to reuse past information. Thus, the agent struggles to
make use of its observation.

This can be addressed in two ways:

- - Change the *model*, to use a permutation-invariant architecture which will not be sensitive to the vehicles order, such as *e.g.* {cite}`Qi2017pointnet` or {cite}`Leurent2019social`.

This example is implemented [here (DQN)](https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/intersection_social_dqn.ipynb) or [here (SB3's PPO)](https://github.com/eleurent/highway-env/blob/master/scripts/sb3_highway_ppo_transformer.py).

- - Change the *observation*. For example, the {ref}`Grayscale Image` does not depend on an ordering. In this case, a CNN model is more suitable than an MLP model.

This example is implemented [here (SB3's DQN)](https://github.com/eleurent/highway-env/blob/master/scripts/sb3_highway_dqn_cnn.py).

#### My videos are too fast / have a low framerate.

This is because in gymnasium, a single video frame is generated at each call of `env.step(action)`. However, in highway-env, the policy typically runs at a low-level frequency (e.g. 1 Hz) so that a long action (*e.g.* change lane) actually corresponds to several (typically, 15) simulation frames.
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
