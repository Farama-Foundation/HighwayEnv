(road-road)=

# Road

A {py:class}`~highway_env.road.road.Road` is composed of a {py:class}`~highway_env.road.road.RoadNetwork` and a list
of {py:class}`~highway_env.vehicle.kinematics.Vehicle`.

The {py:class}`~highway_env.road.road.RoadNetwork` describes the topology of the road infrastructure as a graph,
where edges represent lanes and nodes represent intersections. It contains a {py:attr}`~highway_env.road.road.RoadNetwork.graph` dictionary which stores the {py:class}`~highway_env.road.lane.AbstractLane` geometries by their {py:class}`~highway_env.road.road.LaneIndex`.
A {py:class}`~highway_env.road.road.LaneIndex` is a tuple containing:

- a string identifier of a starting position
- a string identifier of an ending position
- an integer giving the index of the described lane, in the (unique) road from the starting to the ending position

For instance, the geometry of the second lane in the road going from the `"lab"` to the `"pub"` can be obtained by:

```python
lane = road.road_network.graph["lab"]["pub"][1]
```

The actual positions of the lab and the pub are defined in the lane\`\`\`geometry object.

## Neighbour vehicles

Each {py:class}`~highway_env.road.road.Road` exposes
{py:meth}`~highway_env.road.road.Road.neighbour_vehicles` to find the preceding and following vehicles
on a lane. By default, the search is limited to the current lane segment; when
{py:attr}`~highway_env.road.road.Road.neighbour_vehicles_connected_lanes` is enabled, connected next
and previous segments are included. See {ref}`road-neighbour-vehicles` for the full description,
environment version mapping, and a visual comparison.


## PartitionedRoadNetwork
A {py:class}`~highway_env.road.partitioned_road.PartitionedRoadNetwork` is a type of RoadNetwork which supports the partitioning of lanes into grids for significantly faster proximal checks.

Partitioning is done as lanes are added to update a mapping from grid cells to the indices of lanes who pass through them.

The class assumes that lanes will not be removed or be changed in any way after being added.

## API

```{eval-rst}
.. automodule:: highway_env.road.road
    :members:
```
