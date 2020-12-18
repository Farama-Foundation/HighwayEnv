.. _road_road:

Road
#########


A :py:class:`~highway_env.road.road.Road` is composed of a :py:class:`~highway_env.road.road.RoadNetwork` and a list
of :py:class:`~highway_env.vehicle.kinematics.Vehicle`.

The :py:class:`~highway_env.road.road.RoadNetwork` describes the topology of the road infrastructure as a graph,
where edges represent lanes and nodes represent intersections. It contains a :py:attr:`~highway_env.road.road.RoadNetwork.graph` dictionary which stores the :py:class:`~highway_env.road.lane.AbstractLane` geometries by their :py:class:`~highway_env.road.road.LaneIndex`.
A :py:class:`~highway_env.road.road.LaneIndex` is a tuple containing:

- a string identifier of a starting position
- a string identifier of an ending position
- an integer giving the index of the described lane, in the (unique) road from the starting to the ending position

For instance, the geometry of the second lane in the road going from the ``"lab"`` to the ``"pub"`` can be obtained by:

.. code-block:: python

    lane = road.road_network.graph["lab"]["pub"][1]

The actual positions of the lab and the pub are defined in the ``lane```geometry object.

API
*******

.. automodule:: highway_env.road.road
    :members:

