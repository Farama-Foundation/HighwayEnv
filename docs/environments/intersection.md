(environments-intersection)=

```{eval-rst}
.. currentmodule:: highway_env.envs.intersection_env
```

# Intersection

An intersection negotiation task with dense traffic.

```{figure} https://raw.githubusercontent.com/Farama-Foundation/HighwayEnv/gh-media/docs/media/intersection-env.gif
:align: center
:name: fig:intersection_env
:width: 80%
```

```{warning}
It's quite hard to come up with good decentralized behaviors for other agents to avoid each other. Of course, this
could be achieved by sophisticated centralized schedulers, or traffic lights, but to keep things simple a
{ref}`rudimentary collision prediction <road-regulation>` was added in the behaviour of other vehicles.

This simple system sometime fails which results in collisions, blocking the way for the ego-vehicle.
I figured it was fine for my own purpose, since it did not happen too often and it's reasonable to expect
the ego-vehicle to simply wait the end of episode in these situations. But I agree that it is not ideal,
and I welcome any contribution on that matter.

Update (v1.12): this is due to an oversight in neighbour vehicle detection, check
{ref}`road-neighbour-vehicles` for explanation. Many thanks to [@m-walters](https://github.com/m-walters) for reporting this bug and [@Lidang-Jiang](https://github.com/Lidang-Jiang) for proposing a fix.
```

## Usage

```python
env = gym.make("intersection-v0")
```

## Versions

| ID | Description |
|---|---|
| `intersection-v0` | Initial version. Same-segment neighbour search only. Preserved for reproducibility. |
| `intersection-v1` | Continuous-action variant. Same neighbour search as `intersection-v0`. |
| `intersection-v2` | Connected-lane neighbour search enabled by default. Recommended for new experiments. |
| `intersection-multi-agent-v0` | Multi-agent initial version. Same-segment neighbour search only. |
| `intersection-multi-agent-v1` | Multi-agent variant with `MultiAgentWrapper`. Same neighbour search as v0. |
| `intersection-multi-agent-v2` | Multi-agent variant with connected-lane neighbour search enabled. |

See {ref}`road-neighbour-vehicles` for details.

## Default configuration

```python
{
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "absolute": True,
        "flatten": False,
        "observe_intentions": False
    },
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": False,
        "target_speeds": [0, 4.5, 9],
    },
    "duration": 13,  # [s]
    "destination": "o1",
    "controlled_vehicles": 1,
    "initial_vehicle_count": 10,
    "spawn_probability": 0.6,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "collision_reward": -5,
    "high_speed_reward": 1,
    "arrived_reward": 1,
    "reward_speed_range": [7.0, 9.0],
    "normalize_reward": False,
    "offroad_terminal": False,
}
```

More specifically, it is defined in:

```{eval-rst}
.. automethod:: IntersectionEnv.default_config
    :no-index:
```

## API

```{eval-rst}
.. autoclass:: IntersectionEnv
    :members:
```
