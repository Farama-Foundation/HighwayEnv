(environments-intersection)=

```{eval-rst}
.. currentmodule:: highway_env.envs.intersection_env
```

# Intersection

An intersection negotiation task with dense traffic.

```{figure} https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/intersection-env.gif
:align: center
:name: fig:intersection_env
:width: 80%
```

```{warning}
It's quite hard to come up with good decentralized behaviors for other agents to avoid each other. Of course, this
could be achieved by sophisticated centralized schedulers, or traffic lights, but to keep things simple a
{ref}`rudimentary collision prediction <road_regulation>` was added in the behaviour of other vehicles.

This simple system sometime fails which results in collisions, blocking the way for the ego-vehicle.
I figured it was fine for my own purpose, since it did not happen too often and it's reasonable to expect
the ego-vehicle to simply wait the end of episode in these situations. But I agree that it is not ideal,
and I welcome any contribution on that matter.
```

## Usage

```python
env = gym.make("intersection-v0")
```

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
        "longitudinal": False,
        "lateral": True
    },
    "duration": 13,  # [s]
    "destination": "o1",
    "initial_vehicle_count": 10,
    "spawn_probability": 0.6,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "collision_reward": IntersectionEnv.COLLISION_REWARD,
    "normalize_reward": False
}
```

More specifically, it is defined in:

```{eval-rst}
.. automethod:: IntersectionEnv.default_config
```

## API

```{eval-rst}
.. autoclass:: IntersectionEnv
    :members:
```
