.. _environments_merge:

.. currentmodule:: highway_env.envs.merge_env

Merge
**********

In this task, the ego-vehicle starts on a main highway but soon approaches a road junction with incoming vehicles on the access ramp. The agent's objective is now to maintain a high speed while making room for the vehicles so that they can safely merge in the traffic.

.. figure:: https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/merge-env.gif
   :width: 80%
   :align: center
   :name: fig:merge_env

Usage
==========

.. code-block:: python

    env = gym.make("merge-v0")


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
        "simulation_frequency": 15,  # [Hz]
        "policy_frequency": 1,  # [Hz]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 600,  # [px]
        "screen_height": 150,  # [px]
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    }

More specifically, it is defined in:

.. automethod:: MergeEnv.default_config

API
=====

.. autoclass:: MergeEnv
    :members:

