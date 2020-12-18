.. _road_lane:

Lane
#########

The geometry of lanes are described by :py:class:`~highway_env.road.lane.AbstractLane` objects, as a parametrized center line curve, providing a local coordinate system.

Conversions between the (longi, lat) coordinates in the Frenet frame and the global :math:`x,y` coordinates are ensured by the :py:meth:`~highway_env.road.lane.AbstractLane.position` and :py:meth:`~highway_env.road.lane.AbstractLane.local_coordinates` methods.

The main implementations are:

- :py:class:`~highway_env.road.lane.StraightLane`
- :py:class:`~highway_env.road.lane.SineLane`
- :py:class:`~highway_env.road.lane.CircularLane`

API
***

.. automodule:: highway_env.road.lane
    :members:

