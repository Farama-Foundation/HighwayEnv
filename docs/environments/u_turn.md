(environments-u-turn)=

```{eval-rst}
.. currentmodule:: highway_env.envs.u_turn_env
```

# U-Turn

A U-turn risk analysis task: the agent overtakes vehicles that are blocking the traffic while navigating a double-lane U-turn. Six strategically placed vehicles force the agent to balance high-speed overtaking with ensuring safety.

```{figure} ../_static/animations/environments/u-turn-env.gif
:align: center
:name: fig:u_turn_env
:width: 80%
```

## Usage

```python
env = gym.make("u-turn-v0")
```

## Versions

| ID | Description |
|---|---|
| `u-turn-v0` | Initial version. Same-segment neighbour search only. Preserved for reproducibility. |
| `u-turn-v1` | Connected-lane neighbour search enabled by default. Recommended for new experiments. |

See {ref}`road-neighbour-vehicles` for details.

## Default configuration

```python
{
    "observation": {
        "type": "TimeToCollision",
        "horizon": 16,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [8, 16, 24],
    },
    "screen_width": 789,
    "screen_height": 289,
    "duration": 10,  # [s]
    "collision_reward": -1.0,
    "left_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "reward_speed_range": [8, 24],
    "normalize_reward": True,
    "offroad_terminal": False,
}
```

More specifically, it is defined in:

```{eval-rst}
.. automethod:: UTurnEnv.default_config
    :no-index:
```

## API

```{eval-rst}
.. autoclass:: UTurnEnv
    :members:
```
