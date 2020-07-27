.. _make_your_own:

Make your own environment
==========================

Here are the steps required to create a new environment.

.. note::
    Pull requests are welcome!

Set up files
------------

1. Create a new ``your_env.py`` file in ``highway_env/envs/``
2. Define a class YourEnv, that must inherit from :py:class:`~highway_env.envs.common.abstract.AbstractEnv`

This class provides several useful functions:

* A :py:meth:`~highway_env.envs.common.abstract.AbstractEnv.default_config` method, that provides a default configuration dictionary that can be overloaded.
* A :py:meth:`~highway_env.envs.common.abstract.AbstractEnv.define_spaces` method, that gives access to a choice of observation and action types, set from the environment configuration
* A :py:meth:`~highway_env.envs.common.abstract.AbstractEnv.step` method, which executes the desired actions (at policy frequency) and simulate the environment (at simulation frequency)
* A :py:meth:`~highway_env.envs.common.abstract.AbstractEnv.render` method, which renders the environment.

Create the scene
------------------

The first step is to create a :py:class:`~highway_env.road.road.RoadNetwork` that describes the geometry and topology of
roads and lanes in the scene.
This should be achieved in a ``YourEnv._make_road()`` method, called from ``YourEnv.reset()`` to set the ``self.road`` field.

See :ref:`Roads <road_road>` for reference, and existing environments as examples.

Create the vehicles
------------------

The second step is to populate your road network with vehicles. This should be achieved in a ``YourEnv._make_road()``
method, called from ``YourEnv.reset()`` to set the ``self.road.vehicles`` list of :py:class:`~highway_env.vehicle.kinematics.Vehicle`.

First, define the controlled ego-vehicle by setting ``self.vehicle``. The class of controlled vehicle depends on the
choice of action type, and can be accessed as ``self.action_type.vehicle_class``.
Other vehicles can be created more freely, and added to the ``self.road.vehicles`` list.

See :ref:`vehicle behaviors <vehicle_behavior>` for reference, and existing environments as examples.

Make the environment configurable
------------------------------------

To make a part of your environment configurable, overload the :py:meth:`~highway_env.envs.common.abstract.AbstractEnv.default_config`
method to define new ``{"config_key": value}`` pairs with default values. These configurations then be accessed in your
environment implementation with ``self.config["config_key"]``, and once the environment is created, it can be configured with
``env.configure({"config_key": other_value})`` followed by ``env.reset()``.

Register the environment
---------------------------

In ``highway_env/envs/your_env.py``, add the following line:

.. code-block:: python

    register(
        id='your-env-v0',
        entry_point='highway_env.envs:YourEnv',
    )

and import it from ``highway_env/envs/__init__.py``:

.. code-block:: python

    from highway_env.envs.your_env import *


Profit
--------
That's it!
You should now be able to run the environment:

.. code-block:: python

    import gym
    import highway_env

    env = gym.make('your-env-v0')
    obs = env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()

API
-------


.. automodule:: highway_env.envs.common.abstract
    :members:
    :private-members:

