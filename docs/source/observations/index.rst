.. _observations:

.. py:currentmodule::highway_env.envs.common.observation

Observations
=============

For all environments, **several types of observations** can be used. They are defined in the
:py:mod:`~highway_env.envs.common.observation` module.

.. note::
    Each environment comes with a *default* observation in its configuration, which can be changed or parametrised using
    :ref:`environment configurations <configuration>`.

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
            "absolute": false
        }
    })
    env.reset()

.. note::
    The ``"type"`` field in the observation configuration takes values defined in
    :py:func:`~highway_env.envs.common.observation.observation_factory` (see source)

Kinematics
-----------

The :py:class:`~highway_env.envs.common.observation.KinematicObservation` is a :math:`V\times F` array that describes a
list of :math:`V` nearby vehicles by some features of size :math:`F`. For instance:

===========  =========  =========  ===========  ===========
Vehicle      :math:`x`  :math:`y`  :math:`v_x`  :math:`v_y`
===========  =========  =========  ===========  ===========
ego-vehicle  5.0        4.0        15.0          0
vehicle 1    -10.0      4.0        12.0          0
vehicle 2    13.0       8.0        13.5          0
...          ...        ...        ...           ...
vehicle V    22.2       10.5       18.0          0.5
===========  =========  =========  ===========  ===========

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

.. note::
    The ego-vehicle is always described in the first row

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
    The number :math:`V` of vehicles is constant and configured by the ``vehicles_count`` parameter, so that the
    observation has a fixed size. If fewer vehicles than ``vehicles_count`` are observed, the last rows are placeholders
    filled with zeros. The `presence` feature can be used to detect such cases, since it is set to 1 for any observed
    vehicle and 0 for placeholders.



API
--------

.. automodule:: highway_env.envs.common.observation
    :members:
    :inherited-members:
    :show-inheritance:
