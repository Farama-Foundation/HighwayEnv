.. _dynamics:

Dynamics
############

New driving environments can easily be made from a set of building blocks.

Roads
========

A `Road` is composed of a `RoadNetwork` and a list of `Vehicles`. The `RoadNetwork` describes the topology of the road infrastructure as a graph, where edges represent lanes and nodes represent intersections. For every edge, the corresponding lane geometry is stored in a `Lane` object as a parametrized center line curve, providing a local coordinate system.

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