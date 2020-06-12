.. _environments_intersection:

.. currentmodule:: highway_env.envs.intersection_env

Intersection
************

An intersection negotiation task with dense traffic.

.. figure:: https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/intersection-env.gif
   :width: 80%
   :align: center
   :name: fig:intersection_env

Usage
==========

.. code-block:: python

    env = gym.make("intersection-v0")


Default configuration
=====================

.. code-block:: python

    {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20],
            },
            "absolute": True,
            "flatten": False,
            "observe_intentions": False
        },
        "action": {
            "type": "Discrete"
        }
        "duration": 13,  # [s]
        "destination": "o1",
        "initial_vehicle_count": 10,
        "spawn_probability": 0.6,
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.5, 0.6],
        "scaling": 5.5 * 1.3,
        "collision_reward": IntersectionEnv.COLLISION_REWARD,
        "normalize_reward": False
    }

More specifically, it is defined in:

.. automethod:: IntersectionEnv.default_config

API
=====

.. autoclass:: IntersectionEnv
    :members:
    :inherited-members:
    :show-inheritance: