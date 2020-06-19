.. _road_regulation:

Road regulation
#########

A :py:class:`~highway_env.road.regulation.RegulatedRoad` is a :py:class:`~highway_env.road.road.Road` in which the behavior of vehicles take or give the right of way at an intersection based on the :py:attr:`~highway_env.road.lane.AbstractLane.priority` lane attribute.

On such a road, some rules are enforced:

- most of the time, vehicles behave as usual;
- however, they try to predict collisions with other vehicles through the :py:meth:`~highway_env.road.regulation.RegulatedRoad.is_conflict_possible` method;
- when it is the case, right of way is arbitrated through the :py:meth:`~highway_env.road.regulation.RegulatedRoad.respect_priorities` method, and the yielding vehicle target velocity is set to 0 until the conflict is resolved.

API
***

.. automodule:: highway_env.road.regulation
    :members:

