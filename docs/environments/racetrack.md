(environments-racetrack)=

```{eval-rst}
.. currentmodule:: highway_env.envs.racetrack_env
```

# Racetrack

A continuous control environment, where the he agent has to follow the tracks while avoiding collisions with other vehicles.

Credits and many thanks to [@supperted825](https://github.com/supperted825) for the [idea and initial implementation](https://github.com/Farama-Foundation/HighwayEnv/issues/231).

```{figure} https://raw.githubusercontent.com/Farama-Foundation/HighwayEnv/gh-media/docs/media/racetrack-env.gif
:align: center
:name: fig:racetrack_env
:width: 80%
```

## Usage

```python
env = gym.make("racetrack-v0")
```

## Versions

| ID | Description |
|---|---|
| `racetrack-v0` | Initial version. Same-segment neighbour search only. Preserved for reproducibility. |
| `racetrack-v1` | Connected-lane neighbour search enabled by default. Recommended for new experiments. |
| `racetrack-large-v0` | Large three-lane map. Same-segment neighbour search only. |
| `racetrack-large-v1` | Large map with connected-lane neighbour search enabled. |
| `racetrack-oval-v0` | Oval map. Same-segment neighbour search only. |
| `racetrack-oval-v1` | Oval map with connected-lane neighbour search enabled. |

See {ref}`road-neighbour-vehicles` for details.

## Default configuration

```python
{
    "observation": {
        "type": "OccupancyGrid",
        "features": ['presence', 'on_road'],
        "grid_size": [[-18, 18], [-18, 18]],
        "grid_step": [3, 3],
        "as_image": False,
        "align_to_vehicle_axes": True
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": False,
        "lateral": True
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 300,
    "collision_reward": -1,
    "lane_centering_cost": 4,
    "action_reward": -0.3,
    "controlled_vehicles": 1,
    "other_vehicles": 1,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}
```

More specifically, it is defined in:

```{eval-rst}
.. automethod:: RacetrackEnv.default_config
```

## API

```{eval-rst}
.. autoclass:: RacetrackEnv
    :members:
```
