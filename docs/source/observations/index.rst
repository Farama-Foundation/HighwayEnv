.. _observations:

.. py:currentmodule::highway_env.envs.common.observation

Observations
=============

For all environments, **several types of observations** can be used. They are defined in the
:py:mod:`~highway_env.envs.common.observation` module.
Each environment comes with a *default* observation, which can be changed or customised using
:ref:`environment configurations <configuration>`. For instance,

.. code-block:: python

    import gym
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

.. note::
    The ``"type"`` field in the observation configuration takes values defined in
    :py:func:`~highway_env.envs.common.observation.observation_factory` (see source)

Kinematics
-----------

The :py:class:`~highway_env.envs.common.observation.KinematicObservation` is a :math:`V\times F` array that describes a
list of :math:`V` nearby vehicles by a set of features of size :math:`F`, listed in the ``"features"`` configuration field.
For instance:

===========  =========  =========  ===========  ===========
Vehicle      :math:`x`  :math:`y`  :math:`v_x`  :math:`v_y`
===========  =========  =========  ===========  ===========
ego-vehicle  5.0        4.0        15.0          0
vehicle 1    -10.0      4.0        12.0          0
vehicle 2    13.0       8.0        13.5          0
...          ...        ...        ...           ...
vehicle V    22.2       10.5       18.0          0.5
===========  =========  =========  ===========  ===========

.. note::
    The ego-vehicle is always described in the first row

If configured with ``normalized=True`` (default), the observation is normalized within a fixed range, which gives for
the range [100, 100, 20, 20]:

===========  =========  =========  ===========  ===========
Vehicle      :math:`x`  :math:`y`  :math:`v_x`  :math:`v_y`
===========  =========  =========  ===========  ===========
ego-vehicle  0.05       0.04       0.75         0
vehicle 1    -0.1       0.04       0.6          0
vehicle 2    0.13       0.08       0.675        0
...          ...        ...        ...           ...
vehicle V    0.222      0.105      0.9          0.025
===========  =========  =========  ===========  ===========

If configured with ``absolute=False``, the coordinates are relative to the ego-vehicle, except for the ego-vehicle
which stays absolute.


===========  =========  =========  ===========  ===========
Vehicle      :math:`x`  :math:`y`  :math:`v_x`  :math:`v_y`
===========  =========  =========  ===========  ===========
ego-vehicle  0.05       0.04       0.75         0
vehicle 1    -0.15      0          -0.15        0
vehicle 2    0.08       0.04       -0.075       0
...          ...        ...        ...           ...
vehicle V    0.172      0.065      0.15         0.025
===========  =========  =========  ===========  ===========

.. note::
    The number :math:`V` of vehicles is constant and configured by the ``vehicles_count`` field, so that the
    observation has a fixed size. If fewer vehicles than ``vehicles_count`` are observed, the last rows are placeholders
    filled with zeros. The ``presence`` feature can be used to detect such cases, since it is set to 1 for any observed
    vehicle and 0 for placeholders.

Example configuration
~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    import gym
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
    obs = env.reset()
    print(obs)


Grayscale Image
-----------------

The :py:class:`~highway_env.envs.common.observation.GrayscaleObservation` is a :math:`W\times H` grayscale image of the scene, where :math:`W,H` are set with the ``observation_shape`` parameter.
The RGB to grayscale conversion is a weighted sum, configured by the ``weights`` parameter. Several images can be stacked with the ``stack_size`` parameter, as is customary with image observations.

.. warning::
   The ``screen_height`` and ``screen_width`` environment configurations should match the expected ``observation_shape``.

.. warning::
   This observation type required *pygame* rendering, which may be problematic when run on server without display. In this case, the call to ``pygame.display.set_mode()`` raises an exception, which can be avoided by setting the environment configuration ``offscreen_rendering`` to ``True``.

.. _grayscale_example_configuration:

Example configuration
~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    from matplotlib import pyplot as plt
    %matplotlib inline

    screen_width, screen_height = 84, 84
    config = {
        "offscreen_rendering": True,
        "observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "stack_size": 4,
            "observation_shape": (screen_width, screen_height)
        },
        "screen_width": screen_width,
        "screen_height": screen_height,
        "scaling": 1.75,
        "policy_frequency": 2
    }
    env.configure(config)
    obs = env.reset()

    _, axes = plt.subplots(ncols=4, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(obs[..., i], cmap=plt.get_cmap('gray'))
    plt.show()

Illustration of the stack mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We illustrate the stack update by performing three steps in the environment.

.. jupyter-execute::

    for _ in range(3):
        obs, _, _, _ = env.step(env.action_type.actions_indexes["IDLE"])

        _, axes = plt.subplots(ncols=4, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(obs[..., i], cmap=plt.get_cmap('gray'))
    plt.show()

Occupancy grid
---------------

The :py:class:`~highway_env.envs.common.observation.OccupancyGridObservation` is a :math:`W\times H\times F` array,
that represents a grid of shape :math:`W\times H` discretising the space :math:`(X,Y)` around the ego-vehicle in
uniform rectangle cells. Each cell is described by :math:`F` features, listed in the ``"features"`` configuration field.
The grid size and resolution is defined by the ``grid_size`` and ``grid_steps`` configuration fields.

For instance, the channel corresponding to the ``presence`` feature may look like this:

.. table:: presence feature: one vehicle is close to the north, and one is farther to the east.

    ==  ==  ==  ==  ==
    0   0   0   0   0
    0   0   1   0   0
    0   0   0   0   1
    0   0   0   0   0
    0   0   0   0   0
    ==  ==  ==  ==  ==

The corresponding :math:`v_x` feature may look like this:

.. table::  :math:`v_x` feature: the north vehicle drives at the same speed as the ego-vehicle, and the east vehicle a bit slower

    ==  ==  ==  ==  ==
    0   0   0   0   0
    0   0   0   0   0
    0   0   0   0   -0.1
    0   0   0   0   0
    0   0   0   0   0
    ==  ==  ==  ==  ==

Example configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

Time to collision
-----------------


The :py:class:`~highway_env.envs.common.observation.TimeToCollisionObservation` is a :math:`V\times L\times H` array, that represents the predicted time-to-collision of observed vehicles on the same road as the ego-vehicle.
These predictions are performed for :math:`V` different values of the ego-vehicle speed, :math:`L` lanes on the road around the current lane, and represented as one-hot encodings over :math:`H` discretised time values (bins), with 1s steps.

For instance, consider a vehicle at 25m on the right-lane of the ego-vehicle and driving at 15 m/s. Using :math:`V=3,\, L = 3\,H = 10`, with ego-speed of {:math:`15` m/s, :math:`20` m/s and :math:`25` m/s}, the predicted time-to-collisions are :math:`\infty,\,5s,\,2.5s` and the corresponding observation is

==  ==  ==  ==  ==  ==  ==  ==  ==  ==
0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0
==  ==  ==  ==  ==  ==  ==  ==  ==  ==

==  ==  ==  ==  ==  ==  ==  ==  ==  ==
0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0
0   0   0   0   1   0   0   0   0   0
==  ==  ==  ==  ==  ==  ==  ==  ==  ==

==  ==  ==  ==  ==  ==  ==  ==  ==  ==
0   0   0   0   0   0   0   0   0   0
0   0   0   0   0   0   0   0   0   0
0   0   1   0   0   0   0   0   0   0
==  ==  ==  ==  ==  ==  ==  ==  ==  ==

Example configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    "observation": {
        "type": "TimeToCollision"
        "horizon": 10
    },

API
--------

.. automodule:: highway_env.envs.common.observation
    :members:

