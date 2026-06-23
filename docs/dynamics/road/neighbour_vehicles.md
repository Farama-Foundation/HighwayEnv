(road-neighbour-vehicles)=

# Neighbour vehicles

The {py:meth}`~highway_env.road.road.Road.neighbour_vehicles` method finds the preceding and following
vehicles of a given vehicle on a lane. It is used throughout the library — notably by
{py:class}`~highway_env.vehicle.behavior.IDMVehicle` for MOBIL lane-change decisions.
Its accuracy directly affects simulation behaviour in multi-segment environments.

## Problem

Before version 1.12, `neighbour_vehicles()` only searched the **current lane segment**. Vehicles that
are not in the current segment were treated as invisible, even when they were directly ahead of or
behind the ego vehicle in the driving direction.

This caused incorrect behaviour near segment boundaries in environments built from several connected
lanes, such as `merge`, `exit`, `roundabout`, `racetrack`, `intersection`, and `u-turn` maps. See
[issue #626](https://github.com/Farama-Foundation/HighwayEnv/issues/626).

```{figure} ../../_static/img/compare_intersection_v0_v2.gif
:align: center
:width: 100%

Side-by-side comparison of intersection-v0 (left) and intersection-v2 (right).
```

## The fix

When {py:attr}`~highway_env.road.road.Road.neighbour_vehicles_connected_lanes` is enabled, the search
extends to **downstream and upstream connected lane segments** via the road-network graph.

The behaviour is controlled by:

- the `neighbour_vehicles_connected_lanes` flag on {py:class}`~highway_env.road.road.Road`, and
- the matching `neighbour_vehicles_connected_lanes` entry in the environment config (see
  {py:meth}`~highway_env.envs.common.abstract.AbstractEnv.default_config`).

## Reproducibility and environment versions

To preserve reproducibility, existing `*-v0` environment IDs keep the original same-segment neighbour
search by default (`neighbour_vehicles_connected_lanes=False`).

New registered versions enable connected-lane search by default through
{py:class}`~highway_env.envs.common.abstract.ConnectedLaneNeighboursMixin`. For any environment, the
new behaviour can also be enabled explicitly (albeit not recommended):

```python
env = gym.make("merge-v0", config={..., "neighbour_vehicles_connected_lanes": True})
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
connected-lane search) running the same seed and actions side by side. When the ego vehicle passes
a segment boundary, `merge-v1` can detect a rear vehicle that is in the previous segment,
while `merge-v0` cannot. Also notice that vehicle behaviour is **different** even with the same seed.

```{figure} ../../_static/img/compare_merge_v0_v1.gif
:align: center
:width: 100%

Side-by-side comparison of merge-v0 (left) and merge-v1 (right).
```

### Overlay legend

| Overlay | Meaning |
|---|---|
| Green line | Front neighbour returned by `neighbour_vehicles()` |
| Red line | Vehicle that was **not** detected pre-1.12 |
| Blue line | Rear neighbour returned by `neighbour_vehicles()` |
| Yellow line | Lane segment boundary (road-network node) |

## Visual comparison

A pygame program has been created to demonstrate the difference:

[scripts/validate/compare_neighbour_detection.py](https://github.com/Farama-Foundation/HighwayEnv/blob/main/scripts/validate/compare_neighbour_detection.py)

```bash
python scripts/validate/compare_neighbour_detection.py
python scripts/validate/compare_neighbour_detection.py --env merge
python scripts/validate/compare_neighbour_detection.py --env intersection
python scripts/validate/compare_neighbour_detection.py --env racetrack-large --no-patch
python scripts/validate/compare_neighbour_detection.py --validate
python scripts/validate/compare_neighbour_detection.py --fixed-seed --steps 80
```

| Flag | Description |
|---|---|
| `--env` | Environment to compare (default: `merge`) |
| `--no-patch` | Do not patch the left panel with the pre-1.12 implementation |
| `--validate` | Compare current legacy env against patched legacy env with pre-1.12 code |
| `--fixed-seed` | Keep the same seed across loops |
| `--steps` | Number of simulation steps per loop before resetting (default: 80) |

Keyboard controls: **←** rewind, **→** unwind / step forward, **Space** pause / play, **Q** quit.
