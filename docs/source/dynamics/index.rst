.. _dynamics:

Dynamics
############

The dynamics of every environment describes how vehicles move and behave through time.
There are two important sections that affect these dynamics: the description of the roads, and the vehicle physics and behavioral models.

Roads
========


A :py:class:`~highway_env.road.road.Road` is composed of a :py:class:`~highway_env.road.road.RoadNetwork` and a list
of :py:class:`~highway_env.vehicle.kinematics.Vehicle`.

.. toctree::
  :maxdepth: 1

  road/lane
  road/road
  road/regulation

Vehicles
========

.. toctree::
  :maxdepth: 1

  vehicle/kinematics
  vehicle/controller
  vehicle/behavior