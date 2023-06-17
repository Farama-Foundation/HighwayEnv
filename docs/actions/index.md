---
_actions:
---

```{eval-rst}
.. py:module:: highway_env.envs.common.action
```

# Actions

Similarly to {ref}`observations <observations>`, **several types of actions** can be used in every environment. They are defined in the
{py:mod}`~highway_env.envs.common.action` module.
Each environment comes with a *default* action type, which can be changed or customised using
{ref}`environment configurations <configuration>`. For instance,

```python
import gymnasium as gym

env = gym.make('highway-v0', config={
    "action": {
        "type": "ContinuousAction"
    }
})
```

## Continuous Actions

The {py:class}`~highway_env.envs.common.action.ContinuousAction` type allows the agent to directly set the low-level
controls of the {ref}`vehicle kinematics <vehicle_kinematics>`, namely the throttle $a$ and steering angle $\delta$.


```python
import gymnasium as gym

env = gym.make('highway-v0', config={
    "action": {
        "type": "ContinuousAction"
    }
})
```

```{note}
The control of throttle and steering can be enabled or disabled through the
{py:attr}`~highway_env.envs.common.action.ContinuousAction.longitudinal` and {py:attr}`~highway_env.envs.common.action.ContinuousAction.lateral`
configurations, respectively. Thus, the action space can be either 1D or 2D.
```

## Discrete Actions

The {py:class}`~highway_env.envs.common.action.DiscreteAction` is a uniform quantization of the {py:class}`~highway_env.envs.common.action.ContinuousAction` above.

```python
import gymnasium as gym

env = gym.make('highway-v0', config={
    "action": {
        "type": "DiscreteAction"
    }
})
```

The {py:attr}`~highway_env.envs.common.action.DiscreteAction.actions_per_axis` parameter allows to set the quantization step. Similarly to continuous actions, the longitudinal and lateral axis can be enabled or disabled separately.

## Discrete Meta-Actions

The {py:class}`~highway_env.envs.common.action.DiscreteMetaAction` type adds a layer of {ref}`speed and steering controllers <vehicle_controller>`
on top of the continuous low-level control, so that the ego-vehicle can automatically follow the road at a desired velocity.
Then, the available **meta-actions** consist in *changing the target lane and speed* that are used as setpoints for the low-level controllers.

```python
import gymnasium as gym

env = gym.make('highway-v0', config={
    "action": {
        "type": "DiscreteMetaAction"
    }
})
```

The full corresponding action space is defined in {py:attr}`~highway_env.envs.common.action.DiscreteMetaAction.ACTIONS_ALL`

```python
ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
```

Some of these actions might not be always available (lane changes at the edges of the roads, or accelerating/decelrating
beyond the maximum/minimum velocity), and the list of available actions can be accessed with {py:meth}`~highway_env.envs.common.abstract.AbstractEnv.get_available_actions` method.
Taking an unavailable action is equivalent to taking the `IDLE` action.

Similarly to continuous actions, the longitudinal (speed changes) and lateral (lane changes) actions can be disabled separately
through the {py:attr}`~highway_env.envs.common.action.DiscreteMetaAction.longitudinal` and {py:attr}`~highway_env.envs.common.action.DiscreteMetaAction.lateral` parameters.
For instance, in the default configuration of the {ref}`intersection <environments_intersection>` environment, only the speed is controlled by the agent,
while the lateral control of the vehicle is automatically performed by a {ref}`steering controller <vehicle_controller>` to track a desired lane.

## Manual control

The environments can be used as a simulation:

```python
env = gym.make("highway-v0", config={
    "manual_control": True
})
env.reset()
done = False
while not done:
    env.step(env.action_space.sample())  # with manual control, these actions are ignored
```

The ego-vehicle is controlled by directional arrows keys, as defined in
{py:class}`~highway_env.envs.common.graphics.EventHandler`

## API

```{eval-rst}
.. automodule:: highway_env.envs.common.action
    :members:
```
