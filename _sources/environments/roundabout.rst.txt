.. _environments_roundabout:

.. currentmodule:: highway_env.envs.roundabout_env

Roundabout
**********

In this task, the ego-vehicle if approaching a roundabout with flowing traffic. It will follow its planned route automatically, but has to handle lane changes and longitudinal control to pass the roundabout as fast as possible while avoiding collisions.

.. figure:: https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/roundabout-env.gif
   :width: 80%
   :align: center
   :name: fig:roundabout_env

Usage
==========

.. code-block:: python

    env = gym.make("roundabout-v0")


Default configuration
=====================

.. code-block:: python

    {
        "observation": {
            "type": "TimeToCollision"
        },
        "action": {
            "type": "DiscreteMetaAction"
        },
        "incoming_vehicle_destination": None,
        "duration": 11
        "simulation_frequency": 15,  # [Hz]
        "policy_frequency": 1,  # [Hz]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 600,  # [px]
        "screen_height": 600,  # [px]
        "centering_position": [0.5, 0.6],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    }

More specifically, it is defined in:

.. automethod:: RoundaboutEnv.default_config

API
=====

.. autoclass:: RoundaboutEnv
    :members:
