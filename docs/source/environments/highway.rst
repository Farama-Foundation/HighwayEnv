.. _environments_highway:

.. currentmodule:: highway_env.envs.highway_env

Highway
**********

In this task, the ego-vehicle is driving on a multilane highway populated with other vehicles. The agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles. Driving on the right side of the road is also rewarded.

.. figure:: https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/highway.gif
   :width: 80%
   :align: center
   :name: fig:highway_env

Usage
==========

.. code-block:: python

    env = gym.make("highway-v0")


Default configuration
=====================

.. code-block:: python

    {
        "observation": {
            "type": "Kinematics"
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 4,
        "vehicles_count": 50,
        "duration": 40,  # [s]
        "initial_spacing": 2,
        "collision_reward": -1  # The reward received when colliding with a vehicle.
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

.. automethod:: HighwayEnv.default_config

API
=====

.. autoclass:: HighwayEnv
    :members:
