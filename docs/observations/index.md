(observations)=

% py:currentmodule::highway_env.envs.common.observation

# Observations

For all environments, **several types of observations** can be used. They are defined in the
{py:mod}`~highway_env.envs.common.observation` module.
Each environment comes with a *default* observation, which can be changed or customised using
{ref}`environment configurations <configuration>`. For instance,

```python
import gymnasium as gym
import highway_env

env = gym.make('highway-v0')
env.configure({
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
        "grid_step": [5, 5],
        "absolute": False
    }
})
env.reset()
```

```{note}
The `"type"` field in the observation configuration takes values defined in
{py:func}`~highway_env.envs.common.observation.observation_factory` (see source)
```

## Kinematics

The {py:class}`~highway_env.envs.common.observation.KinematicObservation` is a $V\times F$ array that describes a
list of $V$ nearby vehicles by a set of features of size $F$, listed in the `"features"` configuration field.
For instance:

| Vehicle     | $x$   | $y$  | $v_x$ | $v_y$ |
| ----------- | ----- | ---- | ----- | ----- |
| ego-vehicle | 5.0   | 4.0  | 15.0  | 0     |
| vehicle 1   | -10.0 | 4.0  | 12.0  | 0     |
| vehicle 2   | 13.0  | 8.0  | 13.5  | 0     |
| ...         | ...   | ...  | ...   | ...   |
| vehicle V   | 22.2  | 10.5 | 18.0  | 0.5   |

```{note}
The ego-vehicle is always described in the first row
```

If configured with `normalize=True` (default), the observation is normalized within a fixed range, which gives for
the range \[100, 100, 20, 20\]:

| Vehicle     | $x$   | $y$   | $v_x$ | $v_y$ |
| ----------- | ----- | ----- | ----- | ----- |
| ego-vehicle | 0.05  | 0.04  | 0.75  | 0     |
| vehicle 1   | -0.1  | 0.04  | 0.6   | 0     |
| vehicle 2   | 0.13  | 0.08  | 0.675 | 0     |
| ...         | ...   | ...   | ...   | ...   |
| vehicle V   | 0.222 | 0.105 | 0.9   | 0.025 |

If configured with `absolute=False`, the coordinates are relative to the ego-vehicle, except for the ego-vehicle
which stays absolute.

| Vehicle     | $x$   | $y$   | $v_x$  | $v_y$ |
| ----------- | ----- | ----- | ------ | ----- |
| ego-vehicle | 0.05  | 0.04  | 0.75   | 0     |
| vehicle 1   | -0.15 | 0     | -0.15  | 0     |
| vehicle 2   | 0.08  | 0.04  | -0.075 | 0     |
| ...         | ...   | ...   | ...    | ...   |
| vehicle V   | 0.172 | 0.065 | 0.15   | 0.025 |

```{note}
The number $V$ of vehicles is constant and configured by the `vehicles_count` field, so that the
observation has a fixed size. If fewer vehicles than `vehicles_count` are observed, the last rows are placeholders
filled with zeros. The `presence` feature can be used to detect such cases, since it is set to 1 for any observed
vehicle and 0 for placeholders.
```

| Feature      | Description                                                         |
| ------------ | ------------------------------------------------------------------- |
| $presence$   | Disambiguate agents at 0 offset from non-existent agents.           |
| $x$          | World offset of ego vehicle or offset to ego vehicle on the x axis. |
| $y$          | World offset of ego vehicle or offset to ego vehicle on the y axis. |
| $vx$         | Velocity on the x axis of vehicle.                                  |
| $vy$         | Velocity on the y axis of vehicle.                                  |
| $heading$    | Heading of vehicle in radians.                                      |
| $cos_h$      | Trigonometric heading of vehicle.                                   |
| $sin_h$      | Trigonometric heading of vehicle.                                   |
| $cos_d$      | Trigonometric direction to the vehicle's destination.               |
| $sin_d$      | Trigonometric direction to the vehicle's destination.               |
| $long_{off}$ | Longitudinal offset to closest lane.                                |
| $lat_{off}$  | Lateral offset to closest lane.                                     |
| $ang_{off}$  | Angular offset to closest lane.                                     |

### Example configuration

```{eval-rst}
.. jupyter-execute::

    import gymnasium as gym
    import highway_env

    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted"
        }
    }
    env = gym.make('highway-v0')
    env.configure(config)
    obs, info = env.reset()
    print(obs)

```

(grayscale-image)=
## Grayscale Image

The {py:class}`~highway_env.envs.common.observation.GrayscaleObservation` is a $W\times H$ grayscale image of the scene, where $W,H$ are set with the `observation_shape` parameter.
The RGB to grayscale conversion is a weighted sum, configured by the `weights` parameter. Several images can be stacked with the `stack_size` parameter, as is customary with image observations.

(grayscale-example-configuration)=

### Example configuration

```{eval-rst}
.. jupyter-execute::

    from matplotlib import pyplot as plt
    %matplotlib inline
 config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
        "policy_frequency": 2
    }
    env.configure(config)
    obs, info = env.reset()

    fig, axes = plt.subplots(ncols=4, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(obs[i, ...].T, cmap=plt.get_cmap('gray'))
    plt.show()
```

### Illustration of the stack mechanism

We illustrate the stack update by performing three steps in the environment.

```{eval-rst}
.. jupyter-execute::

    for _ in range(3):
        obs, reward, done, truncated, info = env.step(env.action_type.actions_indexes["IDLE"])

        fig, axes = plt.subplots(ncols=4, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(obs[i, ...].T, cmap=plt.get_cmap('gray'))
    plt.show()
```

## Occupancy grid

The {py:class}`~highway_env.envs.common.observation.OccupancyGridObservation` is a $W\times H\times F$ array,
that represents a grid of shape $W\times H$ discretising the space $(X,Y)$ around the ego-vehicle in
uniform rectangle cells. Each cell is described by $F$ features, listed in the `"features"` configuration field.
The grid size and resolution is defined by the `grid_size` and `grid_steps` configuration fields.

For instance, the channel corresponding to the `presence` feature may look like this:

```{eval-rst}
.. table:: presence feature: one vehicle is close to the north, and one is farther to the east.

    ==  ==  ==  ==  ==
    0   0   0   0   0
    0   0   1   0   0
    0   0   0   0   1
    0   0   0   0   0
    0   0   0   0   0
    ==  ==  ==  ==  ==
```

The corresponding $v_x$ feature may look like this:

```{eval-rst}
.. table::  :math:`v_x` feature: the north vehicle drives at the same speed as the ego-vehicle, and the east vehicle a bit slower

    ==  ==  ==  ==  ==
    0   0   0   0   0
    0   0   0   0   0
    0   0   0   0   -0.1
    0   0   0   0   0
    0   0   0   0   0
    ==  ==  ==  ==  ==
```

### Example configuration

```python
"observation": {
    "type": "OccupancyGrid",
    "vehicles_count": 15,
    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    "features_range": {
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20]
    },
    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
    "grid_step": [5, 5],
    "absolute": False
}
```

## Time to collision

The {py:class}`~highway_env.envs.common.observation.TimeToCollisionObservation` is a $V\times L\times H$ array, that represents the predicted time-to-collision of observed vehicles on the same road as the ego-vehicle.
These predictions are performed for $V$ different values of the ego-vehicle speed, $L$ lanes on the road around the current lane, and represented as one-hot encodings over $H$ discretised time values (bins), with 1s steps.

For instance, consider a vehicle at 25m on the right-lane of the ego-vehicle and driving at 15 m/s. Using $V=3,\, L = 3\,H = 10$, with ego-speed of {$15$ m/s, $20$ m/s and $25$ m/s}, the predicted time-to-collisions are $\infty,\,5s,\,2.5s$ and the corresponding observation is

```{eval-rst}
.. table:: $15$ m/s

==  ==  ==  ==  ==  ==  ==  ==  ==  ==  
0   0   0   0   0   0   0   0   0   0  
0   0   0   0   0   0   0   0   0   0  
0   0   0   0   0   0   0   0   0   0  
==  ==  ==  ==  ==  ==  ==  ==  ==  ==
```

```{eval-rst}
.. table:: $20$ m/s

==  ==  ==  ==  ==  ==  ==  ==  ==  ==
0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0
0   0   0   0   1   0   0   0   0   0
==  ==  ==  ==  ==  ==  ==  ==  ==  ==
```

```{eval-rst}
.. table:: $25$ m/s

==  ==  ==  ==  ==  ==  ==  ==  ==  ==
0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0
0   0   1   0   0   0   0   0   0   0
==  ==  ==  ==  ==  ==  ==  ==  ==  ==
```

The top row corresponds to the left-lane, the middle row corresponds to the lane where ego-vehicle is located, and the bottom row to the right-lane.

### Example configuration

```python
"observation": {
    "type": "TimeToCollision"
    "horizon": 10
},
```

## Lidar

The {py:class}`~highway_env.envs.common.observation.LidarObservation` divides the space around the vehicle into angular sectors, and returns an array with one row per angular sector and two columns:
  - distance to the nearest collidable object (vehicles or obstacles)
  - component of the objects's relative velocity along that direction 

The angular sector of index 0 corresponds to an angle 0 (east), and then each index/sector increases the angle (south, west, north). 

For example, for a grid of 8 cells, an obstacle 10 meters away in the south and moving towards the north at 1m/s would lead to the following observation:
    
```{eval-rst}
.. table:: the Lidar observation 

    ===   ===
    0     0 
    0     0 
    10    -1
    0     0
    0     0
    0     0
    0     0
    0     0
    ===   ===
```

Here is an example of what the distance grid may look like in the parking env:

```{eval-rst}
.. jupyter-execute::

    env = gym.make(
        'parking-v0',
        render_mode='rgb_array',
        config={
            "observation": {
                "type": "LidarObservation",
                "cells": 128,
            },
            "vehicles_count": 3,
        })
    env.reset()
    
    plt.imshow(env.render())
    plt.show()
```

You can configure the number of cells in the angular grid with the `cells` parameter, the maximum range with `maximum_range`, and if you enable `normalize`, then distances and relative speeds are both divided by the maximum range.


### Example configuration

```python
"observation": {
    "type": "LidarObservation",
    "cells": 128,
    "maximum_range": 64,
    "normalise": True,
}
```

## API

```{eval-rst}
.. automodule:: highway_env.envs.common.observation
    :members:
```
