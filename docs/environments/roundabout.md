(environments-roundabout)=

```{eval-rst}
.. currentmodule:: highway_env.envs.roundabout_env
```

# Roundabout

In this task, the ego-vehicle if approaching a roundabout with flowing traffic. It will follow its planned route automatically, but has to handle lane changes and longitudinal control to pass the roundabout as fast as possible while avoiding collisions.

```{figure} https://raw.githubusercontent.com/Farama-Foundation/HighwayEnv/gh-media/docs/media/roundabout-env.gif
:align: center
:name: fig:roundabout_env
:width: 80%
```

## Usage

```python
env = gym.make("roundabout-v0")
```

## Versions

| ID | Description |
|---|---|
| `roundabout-v0` | Initial version. Same-segment neighbour search only. Preserved for reproducibility. |
| `roundabout-v1` | Connected-lane neighbour search enabled by default. Recommended for new experiments. |
| `roundabout-generic-v0` | Initial generic roundabout map. Same-segment neighbour search only. |
| `roundabout-generic-v1` | Generic roundabout map with connected-lane neighbour search enabled. |

See {ref}`road-neighbour-vehicles` for details.

## Default configuration

```python
{
    "observation": {
        "type": "Kinematics",
        "absolute": True,
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-15, 15],
            "vy": [-15, 15],
        },
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [0, 8, 16],
    },
    "incoming_vehicle_destination": None,
    "collision_reward": -1,
    "high_speed_reward": 0.2,
    "right_lane_reward": 0,
    "lane_change_reward": -0.05,
    "normalize_reward": True,
    "duration": 11,  # [s]
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 600,  # [px]
    "centering_position": [0.5, 0.6],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}
```

More specifically, it is defined in:

```{eval-rst}
.. automethod:: RoundaboutEnv.default_config
```

## API

```{eval-rst}
.. autoclass:: RoundaboutEnv
    :members:
```
