(graphics)=

% py:currentmodule::highway_env.envs.common.graphics

# Graphics

Environment rendering is done with [pygame](https://www.pygame.org/news), which must be {ref}`installed separately <installation>`.

A window is created at the first call of `env.render()`. Its dimensions can be configured:

```python
env = gym.make("roundabout-v0")
env.configure({
    "screen_width": 640,
    "screen_height": 480
})
env.reset()
env.render()
```

## World surface

The simulation is rendered in a {py:class}`~highway_env.envs.common.graphics.RoadSurface` pygame surface, which defines the location and zoom of the rendered location.
By default, the rendered area is always centered on the ego-vehicle.
Its initial scale and offset can be set with the `"scaling"` and `"centering_position"` configurations, and can also be
updated during simulation using the O,L keys and K,M keys, respectively.

## Scene graphics

- Roads are rendered in the {py:class}`~highway_env.road.graphics.RoadGraphics` class.
- Vehicles are rendered in the {py:class}`~highway_env.vehicle.graphics.VehicleGraphics` class.

## API

```{eval-rst}
.. automodule:: highway_env.envs.common.graphics
    :members:
```

```{eval-rst}
.. automodule:: highway_env.road.graphics
    :members:
```

```{eval-rst}
.. automodule:: highway_env.vehicle.graphics
    :members:
```
