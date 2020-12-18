.. _environments_parking:

.. currentmodule:: highway_env.envs.parking_env

Parking
**********

A goal-conditioned continuous control task in which the ego-vehicle must park in a given space with the appropriate heading.

.. figure:: https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/parking-env.gif
   :width: 80%
   :align: center
   :name: fig:parking_env

Usage
==========

.. code-block:: python

    env = gym.make("parking-v0")


Default configuration
=====================

.. code-block:: python

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
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "screen_width": 600,
        "screen_height": 300,
        "centering_position": [0.5, 0.5],
        "scaling": 7
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    }

More specifically, it is defined in:

.. automethod:: ParkingEnv.default_config

API
=====

.. autoclass:: ParkingEnv
    :members:
