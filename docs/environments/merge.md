(environments-merge)=

```{eval-rst}
.. currentmodule:: highway_env.envs.merge_env
```

# Merge

In this task, the ego-vehicle starts on a main highway but soon approaches a road junction with incoming vehicles on the access ramp. The agent's objective is now to maintain a high speed while making room for the vehicles so that they can safely merge in the traffic.

```{figure} https://raw.githubusercontent.com/Farama-Foundation/HighwayEnv/gh-media/docs/media/merge-env.gif
:align: center
:name: fig:merge_env
:width: 80%
```

## Usage

```python
env = gym.make("merge-v0")
```

## Versions

| ID | Description |
|---|---|
| `merge-v0` | Initial version. Same-segment neighbour search only (`neighbour_vehicles_connected_lanes=False`). Preserved for reproducibility. |
| `merge-v1` | Connected-lane neighbour search enabled by default. Recommended for new experiments. |

See {ref}`road-neighbour-vehicles` for details.

## Default configuration

```python
{
    "observation": {
        "type": "TimeToCollision"
    },
    "action": {
        "type": "DiscreteMetaAction"
    },
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}
```

More specifically, it is defined in:

```{eval-rst}
.. automethod:: MergeEnv.default_config
```

## API

```{eval-rst}
.. autoclass:: MergeEnv
    :members:
```
