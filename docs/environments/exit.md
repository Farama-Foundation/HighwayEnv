(environments-exit)=

```{eval-rst}
.. currentmodule:: highway_env.envs.exit_env
```

# Exit

In this task, the ego-vehicle is driving on a multilane highway and must navigate across lanes to reach an exit ramp. The agent's objective is to successfully take the exit while maintaining speed and avoiding collisions with surrounding traffic.

```{figure} ../_static/animations/environments/exit-env.gif
:align: center
:name: fig:exit_env
:width: 80%
```

## Usage

```python
env = gym.make("exit-v2")
```

## Versions

| ID | Description |
|---|---|
| `exit-v0` | Initial version. Same-segment neighbour search only. Preserved for reproducibility. |
| `exit-v1` | Connected-lane neighbour search enabled by default. |
| `exit-v2` | Same as `exit-v1`, but the reward depends on actually taking the exit. Recommended for new experiments. |

See {ref}`road-neighbour-vehicles` for details.

## Default configuration

```python
{
    "observation": {
        "type": "ExitObservation",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "clip": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [18, 24, 30],
    },
    "lanes_count": 6,
    "collision_reward": 0,
    "high_speed_reward": 0.1,
    "right_lane_reward": 0,
    "normalize_reward": True,
    "goal_reward": 1,
    "vehicles_count": 20,
    "vehicles_density": 1.5,
    "controlled_vehicles": 1,
    "duration": 18,  # [s]
    "simulation_frequency": 5,
    "scaling": 5,
}
```

In `exit-v2`, the episode ends when the ego-vehicle takes the exit, crashes, or drives past the exit, and the following keys are changed:

```python
{
    "collision_reward": -1,
    "goal_reward": 1,  # Given once, when the ego-vehicle is on the exit ramp.
    "missed_exit_reward": -1,  # Driving past the exit, or running out of time before taking it.
    "high_speed_reward": 0.02,
    "lane_progress_reward": 0.5,  # Split over the lane changes towards the exit lane, taken back when moving away from it.
    "normalize_reward": False,
}
```

More specifically, it is defined in:

```{eval-rst}
.. automethod:: ExitEnv.default_config
    :no-index:
```

## API

```{eval-rst}
.. autoclass:: ExitEnv
    :members:

.. autoclass:: ExitEnvV2
    :members:
```
