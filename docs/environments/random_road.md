(environments-random-road)=

```{eval-rst}
.. currentmodule:: highway_env.envs.random_road_env
```

# Random Road
Agent has low-level controls of a car and needs to navigate a procedurally generated road network to a goal parking spot.

The rules are minimal:
1. Get to your destination in as little time as possible
2. Don't crash

```{figure} ../_static/animations/environments/random-road.gif
:align: center
:name: fig:random_road_env
:width: 80%
```


## Usage

```python
env = gym.make("random-road-v0")
env.reset()
```
Custom generation parameters are passed via `config`. A deterministic seed can also be specified with env.reset:
<br>
```python
generation_params =  {
    "target_num_endpoints": 50,
    "forward_speed": 10,
    "age_of_maturity": 4,
    "lane_width": 15,
    "perlin_variation_params": {
        "jitteriness": {"upper": 0.1, "lower": 0.0},
        "max_turn_speed": {"upper": 4.0, "lower": 0.01},
        "replication_chance": {"upper": 0.7, "lower": 0.0},
        "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
    },
    "disable_prints": False,
}
env = gym.make("random-road-v0")
config = {"generation_params": generation_params}
env.reset(seed = 0, options = {'config': config})
```

```{note}
It is recommended to pass the generation parameters via `options['config']` on calling reset and not at initialization. This is because, during initialization, `env.reset` is internally called and generates a road network (before being undone by the first external `env.reset` call). If you decide to provide your own generation parameters under config in `gym.make`, the vestigal generation call will take an unnecessarily longer amount of time compared to the automatically provided default parameters, which has a `target_num_endpoints` of only 5 and should finish instantly without significant overhead.
```
Additionally, pre-existing lanes may be saved for reuse to prevent the unnecessary overhead of generating the same road network from scratch every round.
To do this, use `config['preloaded_lanes']` This will overrule your choice of seed and generation parameters.
```python
from highway_env.road.generation.generator import {
    save_lanes_to_disk,
    load_lanes_from_disk,
}

env = gym.make("random-road-v0")
env.reset()
save_lanes_to_disk("lanes.npz", env.unwrapped.lanes)

[...]

preloaded_lanes = load_lanes_from_disk("lanes.npz")
config = {"preloaded_lanes": preloaded_lanes}
env.reset(options = {'config': config})
```

## Versions

| ID | Description |
|---|---|
| `u-turn-v0` | Initial version. Single-agent only.|

## Default configuration

```python
{
    "observation": {
        "type": "TupleObservation",
        "observation_configs": [
            {"type": "LaneLidarObservation"},
            {"type": "NavigationObservation"},
            {"type": "RelativeGoalObservation"},
            {"type": "LidarObservation"},
        ],
    },
    "action": {"type": "ContinuousAction"},
    "screen_width": 1200,
    "screen_height": 700,
    "max_timesteps": 1000,
    "curb_collision_reward": -10,
    "car_collision_reward": -20,
    "parking_reward": 10,
    "parking_score_threshold": 0.7,
    "parking_score_weights": [0.5, 1, 3],
    "route_following_reward_scalar": 0.1,
    "timestep_reward": -0.01,
    "parking_seed": 0,
    "generation_params": None,
    "preloaded_lanes": None,
    "lane_partition_gridsize": 100,
}
```

More specifically, it is defined in:

```{eval-rst}
.. automethod:: RandomRoadEnv.default_config
    :no-index:
```

## Rewards
- Curb collision penalty
- Vehicle-Vehicle collision penalty
- Parking reward
- Timestep punishment (incentivizes speed)
- Route-following reward (scalar projection of velocity onto navigation arrow)


## Termination & Truncation
Termination occurs when an agent either crashes or parks successfully.
Truncation will occur after a fixed number of timesteps.


## API

```{eval-rst}
.. autoclass:: RandomRoadEnv
    :members:
```
