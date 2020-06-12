.. _vehicle_behavior:

Behavior
==========

.. py:module::highway_env.vehicle.behavior
.. py:currentmodule::highway_env.vehicle.behavior

The vehicles populating the highway follow simple and realistic behaviours that dictate how they accelerate and
steer on the road.

In the :py:class:`IDMVehicle` class,

* Longitudinal Model: the acceleration of the vehicle is given by the Intelligent Driver Model (IDM) from :cite:`Treiber2000`.
* Lateral Model: the discrete lane change decisions are given by the MOBIL model from


In the :py:class:`highway_env.vehicle.behavior.LinearVehicle` class, the longitudinal and lateral behaviours
are defined as linear weightings of several features, such as the distance and speed difference to the leading
vehicle.


API
***

.. automodule:: highway_env.vehicle.behavior
    :members:
    :inherited-members:
    :show-inheritance:
