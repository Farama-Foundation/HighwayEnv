(environments-two-way)=

```{eval-rst}
.. currentmodule:: highway_env.envs.two_way_env
```

# Two Way

A risk management task: the agent is driving on a two-way road with oncoming traffic. It must balance making progress by overtaking slower vehicles and ensuring safety. These conflicting objectives are implemented by a reward signal and a constraint signal, in the CMDP/BMDP framework.

```{figure} https://raw.githubusercontent.com/Farama-Foundation/HighwayEnv/gh-media/docs/media/two-way-env.gif
:align: center
:name: fig:two_way_env
:width: 80%
```

## Usage

```python
env = gym.make("two-way-v0")
```

## Default configuration

```python
{
    "observation": {
        "type": "TimeToCollision",
        "horizon": 5,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "collision_reward": 0,
    "left_lane_constraint": 1,
    "left_lane_reward": 0.2,
    "high_speed_reward": 0.8,
}
```

More specifically, it is defined in:

```{eval-rst}
.. automethod:: TwoWayEnv.default_config
    :no-index:
```

## API

```{eval-rst}
.. autoclass:: TwoWayEnv
    :members:
```
