(environments-parking)=

```{eval-rst}
.. currentmodule:: highway_env.envs.parking_env
```

# Parking

A goal-conditioned continuous control task in which the ego-vehicle must park in a given space with the appropriate heading.

```{figure} ../_static/animations/environments/parking-env.gif
:align: center
:name: fig:parking_env
:width: 80%
```

## Usage

```python
env = gym.make("parking-v0")
```

## Versions

| ID | Description |
|---|---|
| `parking-v0` | Default parking task with an empty lot. |
| `parking-ActionRepeat-v0` | Lower policy frequency (1 Hz) and shorter episode (20 s). |
| `parking-parked-v0` | Lot populated with 10 parked vehicles as obstacles. |

## Default configuration

```python
{
    "observation": {
        "type": "KinematicsGoal",
        "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False
    },
    "action": {
        "type": "ContinuousAction"
    },
    "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
    "success_goal_reward": 0.12,
    "collision_reward": -5,
    "steering_range": np.deg2rad(45),
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 100,
    "controlled_vehicles": 1,
    "vehicles_count": 0,
    "add_walls": True,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": None
}
```

More specifically, it is defined in:

```{eval-rst}
.. automethod:: ParkingEnv.default_config
    :no-index:
```

## API

```{eval-rst}
.. autoclass:: ParkingEnv
    :members:
```
