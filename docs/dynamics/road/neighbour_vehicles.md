(road-neighbour-vehicles)=

# Neighbour vehicles

The {py:meth}`~highway_env.road.road.Road.neighbour_vehicles` method finds the preceding and following
vehicles of a given vehicle on a lane. It is used throughout the library — notably by
{py:class}`~highway_env.vehicle.behavior.IDMVehicle` for longitudinal control and MOBIL lane-change
decisions — so its accuracy directly affects simulation behaviour in multi-segment environments.

## Problem

Before version 1.12, `neighbour_vehicles()` only searched the **current lane segment**. Vehicles that
had already crossed into a connected next or previous segment were treated as invisible, even when they
were directly ahead of or behind the ego vehicle in the driving direction.

This caused incorrect behaviour near segment boundaries in environments built from several connected
lanes, such as merge, exit, roundabout, racetrack, intersection, and u-turn maps. See
[issue #626](https://github.com/Farama-Foundation/HighwayEnv/issues/626).

## The fix

When {py:attr}`~highway_env.road.road.Road.neighbour_vehicles_connected_lanes` is enabled, the search
extends to **downstream and upstream connected lane segments** via the road-network graph. Longitudinal
coordinates on each connected lane are mapped into the ego lane's coordinate frame using appropriate
offsets (the length of the current lane for next segments, and the negative length of the previous lane
for upstream segments).

The behaviour is controlled by:

- the `neighbour_vehicles_connected_lanes` flag on {py:class}`~highway_env.road.road.Road`, and
- the matching `neighbour_vehicles_connected_lanes` entry in the environment config (see
  {py:meth}`~highway_env.envs.common.abstract.AbstractEnv.default_config`).

## Reproducibility and environment versions

To preserve reproducibility for coursework and published experiments, existing `*-v0` environment IDs
keep the original same-segment neighbour search by default
(`neighbour_vehicles_connected_lanes=False`).

New registered versions enable connected-lane search by default through
{py:class}`~highway_env.envs.common.abstract.ConnectedLaneNeighboursMixin`. For any environment, the
new behaviour can also be enabled explicitly (albeit not recommended):

```python
env = gym.make("merge-v0", config={"neighbour_vehicles_connected_lanes": True})
```

### Version mapping

| Environment | Initial (legacy search) | Connected-lane search |
|---|---|---|
| exit | `exit-v0` | `exit-v1` |
| merge | `merge-v0` | `merge-v1` |
| roundabout | `roundabout-v0`, `roundabout-generic-v0` | `roundabout-v1`, `roundabout-generic-v1` |
| racetrack | `racetrack-v0`, `racetrack-large-v0`, `racetrack-oval-v0` | `racetrack-v1`, `racetrack-large-v1`, `racetrack-oval-v1` |
| u-turn | `u-turn-v0` | `u-turn-v1` |
| intersection | `intersection-v0`, `intersection-multi-agent-v0` | `intersection-v2`, `intersection-multi-agent-v2` |

```{note}
`intersection-v1` and `intersection-multi-agent-v1` are unrelated to this change: they provide a
continuous-action variant and a multi-agent wrapper respectively, not connected-lane neighbour search.
```

## Demonstration

The animation below compares `merge-v0` (left, original same-segment search) with `merge-v1` (right,
connected-lane search) running the same seed and actions side by side. When the ego vehicle approaches
a segment boundary, `merge-v1` can detect a lead vehicle that has already entered the next segment,
while `merge-v0` cannot. Also notice that vehicle behaviour is different even with the same seed.

```{figure} ../../../../_static/img/compare_merge_v0_v1.gif
:align: center
:width: 100%

Side-by-side comparison of merge-v0 (left) and merge-v1 (right).
```

### Overlay legend

| Overlay | Meaning |
|---|---|
| Green solid line | Front neighbour returned by `neighbour_vehicles()` |
| Red dashed line | Vehicle on the next connected segment that was **not** detected |
| Blue solid line | Rear neighbour returned by `neighbour_vehicles()` |
| Yellow dashed line | Lane segment boundary (road-network node) |

## Interactive comparison

A pygame script is provided to explore the difference interactively:

[scripts/compare_merge_v0_v1.py](https://github.com/Farama-Foundation/HighwayEnv/blob/main/scripts/compare_merge_v0_v1.py)

```bash
python scripts/compare_merge_v0_v1.py
python scripts/compare_merge_v0_v1.py --no-patch
python scripts/compare_merge_v0_v1.py --validate
python scripts/compare_merge_v0_v1.py --fixed-seed --steps 80
```

| Flag | Description |
|---|---|
| `--no-patch` | Do not patch the left panel with the pre-1.12 implementation |
| `--validate` | Compare current `merge-v0` against patched `merge-v0` with pre-1.12 code |
| `--fixed-seed` | Keep the same seed across loops instead of incrementing it |
| `--steps` | Number of simulation steps per loop before resetting (default: 80) |

Keyboard controls: **←** rewind, **→** unwind / step forward, **Space** pause/play, **Q** quit.

Pygame is required to run the viewer.
