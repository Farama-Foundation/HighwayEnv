(graphics)=

% py:currentmodule::highway_env.envs.common.graphics

# Graphics

Environment rendering is done with [pygame community edition](https://pyga.me), a maintained fork of the original [pygame](https://www.pygame.org/news) library.

## Render modes

```{warning}
Since v1.12, `offscreen_rendering` defaults to `None` and is derived from `render_mode`.
Previously, users would manually set `offscreen_rendering=True` for headless rendering.
The associated `OFFSCREEN_RENDERING` environment variable is also deprecated & ignored.
```

HighwayEnv follows [Gymnasium's convention](https://gymnasium.farama.org/main/api/env/#gymnasium.Env.render) for `render_mode`:

```python
import gymnasium as gym

# No rendering (fastest, for training with no visual information)
env = gym.make("highway-v0")

# Pixel array (for recording or pixel-based observations)
env = gym.make("highway-v0", render_mode="rgb_array")

# Visual window (for human viewing, render as fast as possible)
env = gym.make("highway-v0", render_mode="human")

# Visual window (for human viewing, at pre-set framerate)
env = gym.make("highway-v0", render_mode="human", config={"real_time_rendering": True})
```

| `render_mode`   | Default window | `render()` returns    | Auto-renders on step/reset |
|-----------------|----------------|-----------------------|----------------------------|
| `None`          | No             | `None` (warns)        | No                         |
| `"rgb_array"`   | No             | `np.ndarray (H,W,3)`  | No                         |
| `"human"`       | Yes            | `None`                | Yes                        |

## Rendering configuration

The following config keys control rendering behavior. When `offscreen_rendering` is left as `None` (default), it would be derived from `render_mode`:

| Config key             | Default                                 | Window |
|------------------------|-----------------------------------------|--------|
| `offscreen_rendering`  | `False` for `"human"`, `True` otherwise | No when `True`, Yes when `False` |
| `real_time_rendering`  | `False`                                 | N/A    |

- **`offscreen_rendering`**: When `True`, no pygame display window is created. Drawing is done to an off-screen surface only. Useful for headless servers or when you only need pixel arrays.
- **`real_time_rendering`**: When `True`, the display is synced to render at `simulation_frequency` frames per second so that simulations play at real-time speed. Only has an effect when a window is shown (i.e. `render_mode="human"` or `offscreen_rendering=False`).

These can be explicitly overridden in config when needed:

```python
# rgb_array mode but also show a window (e.g. for debugging)
env = gym.make("highway-v0", render_mode="rgb_array", config={"offscreen_rendering": False})
```

## Window configuration

A window is created at the first call of `env.render()`. Its dimensions can be configured:

```python
env = gym.make(
    "roundabout-v0",
    render_mode="human",
    config={
        "screen_width": 640,
        "screen_height": 480
    }
)
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
