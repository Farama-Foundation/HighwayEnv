(environments-lane-keeping)=

```{eval-rst}
.. currentmodule:: highway_env.envs.lane_keeping_env
```

# Lane Keeping

A pure lateral control task using a bicycle dynamics model. The agent must steer to follow a sine-wave lane with no other traffic. The reward is maximised when the vehicle stays centred on the lane.

## Usage

```python
env = gym.make("lane-keeping-v0")
```

## Default configuration

```python
{
    "observation": {
        "type": "AttributesObservation",
        "attributes": ["state", "derivative", "reference_state"],
    },
    "action": {
        "type": "ContinuousAction",
        "steering_range": [-np.pi / 3, np.pi / 3],
        "longitudinal": False,
        "lateral": True,
        "dynamical": True,
    },
    "simulation_frequency": 10,
    "policy_frequency": 10,
    "state_noise": 0.05,
    "derivative_noise": 0.05,
    "screen_width": 600,
    "screen_height": 250,
    "scaling": 7,
    "centering_position": [0.4, 0.5],
}
```

More specifically, it is defined in:

```{eval-rst}
.. automethod:: LaneKeepingEnv.default_config
    :no-index:
```

## API

```{eval-rst}
.. autoclass:: LaneKeepingEnv
    :members:
```
