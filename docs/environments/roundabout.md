(environments-roundabout)=

```{eval-rst}
.. currentmodule:: highway_env.envs.roundabout_env
```

# Roundabout

In this task, the ego-vehicle if approaching a roundabout with flowing traffic. It will follow its planned route automatically, but has to handle lane changes and longitudinal control to pass the roundabout as fast as possible while avoiding collisions.

```{figure} https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/roundabout-env.gif
:align: center
:name: fig:roundabout_env
:width: 80%
```

## Usage

```python
env = gym.make("roundabout-v0")
```

## Default configuration

```python
{
    "observation": {
        "type": "TimeToCollision"
    },
    "action": {
        "type": "DiscreteMetaAction"
    },
    "incoming_vehicle_destination": None,
    "duration": 11, # [s] If the environment runs for 11 seconds and still hasn't done(vehicle is crashed), it will be truncated. "Second" is expressed as the variable "time", equal to "the number of calls to the step method" / policy_frequency.
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px] width of the pygame window
    "screen_height": 600,  # [px] height of the pygame window
    "centering_position": [0.5, 0.6],  # The smaller the value, the more southeast the displayed area is. K key and M key can change centering_position[0].
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
