# HighwayEnv's Procedural Road Network Generator

Generates arbitrary road networks of any size defined by the raw geometry of various lanes.


## Use

```python
from highway_env.road.generation.generator import
generate_random_lanes, default_params

from highway_env.road.lane import PolyLane
from highway_env.road.partitioned_road import PartitionedRoadNetwork
from highway_env.road.road import LineType, Road

generation_params = default_params()
generation_params['seed'] = 12345

try:
    lanes = generate_random_lanes(generation_params)
except Exception as e:
    raise RuntimeError(
        "Fatal error encountered when generating road network."
        "If this issue persists, try a different seed."
        f"\n\tOriginal error: {e}"
    ) from e

# We now have a list of Lane that contains the raw geometrical boundary points
# of our road network + the nodes each lane connects to.
# We must now construct our RoadNetwork object from this

net = PartitionedRoadNetwork(partition_gridsize=100)

for lane in lanes:
    real_lane = PolyLane(
        lane_points=lane.points,
        left_boundary_points=lane.left_points,
        right_boundary_points=lane.right_points,
        line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
    )
    net.add_lane(lane.start, lane.end, real_lane, bidirectional=True)

self.road = Road(net)
```


## The Generation Process
The generation process occurs in 5 distinct phases:
1. Random Swarm Generation
2. Rectification
3. Twist Optimization
4. Boundary Creation
5. Validation

### Phase 1: Random Swarm Generation
- We start with a single agent at the origin
- Agents move forward, randomly turn left/right, and can spontaneously reproduce, creating a fork with 1-3 offshooting new agents. Agents will die if they ram into another agent or lane.
- Each agent corresponds to one lane. The path an agent takes broadly dictates the path of a lane. Where the agent spawns and where it dies dictate where the endpoints of the lane are.

### Phase 2: 'Rectification'
1. If a lane is too 'young' or short (< 3 points) we must either kill it or add a new point
2. Neighboring endpoints that form a 'junction' must be made to have the same string location identifier
3. Endpoints that ram into other lanes must make a junction with the other lane. This means dividing lanes into two halves at the new junction point
4. Finally, looping lanes whose start and end location are the same are considered useless and are removed.

### Phase 3: Twist Optimization
- The endpoints of multiple lanes at an intersection must properly 'point' into each other. To ensure this happens, we 'twist' the endpoints into an optimal configuration via gradient-descent. Twisting means making each line segment of a lane a 'servo actuator' that is attached to the preceding line segment, and rotating them a fixed amount in unison.
- If the lane is too short to be twisted, we rotate the whole lane
- Endpoints are also 'trimmed' or 'squished' to ensure they don't overshoot the intersection point of other endpoints in the junction.
- Additionally, where there are two lanes that have the same start + end location and trace a similar path after twisting, one is removed.

### Phase 4: Boundary Creation
- Borders are defined by a left and right sequence of points
- At junctions, these borders are made to intersect at common points properly with the borders of other lanes.
- Dead-ends (junctions with only one endpoint) are sealed off as well by letting the right boundary point extend over to the left side.

### Phase 5: Validation
- Mistakes in the generation process, although rare, can still occur, such as
lanes that jutt into other lanes, overly narrow intersection bottlenecks, or even disjoint, unreachable clusters of road network.
- To combat this, every lane is checked for its traversibility. This is done by simulating small physics-based balls that are drawn to the end of the lane and are repelled by its borders. If no ball reaches the end of the lane, the lane is considered blocked. All blocked lanes are then removed from the network.
- The network is also checked for any separate, disjoint clusters. If there are multiple such unreachable clusters, only the largest one is retained and the rest are deleted entirely.


## Parameters Guide
- **target_num_endpoints**: total number of intersections to generate (how big should the road network be?)
- **forward_speed**: step speed of an agent; determines how long each individual lane segment is
- **age_of_maturity**: timesteps before an agent can replicate or die; provides a hard limit on the interval between two consecutive forks.
- **lane_width**:	lateral width of lane.
- **perlin_variation_params** - Certain parameters are allowed to be continually varied across space so as to create differing regions or 'biomes' with differing road properties. For each of these parameters, a lower and upper bound on their value is specified. Here are these such parameters:
    - **jitteriness** - erraticity of how agents turn left/right (Lower value = smoother paths; higher value = more unpredictable)
    - **max_turn_speed** - maximum angular velocity of an agent (lower = straighter roads)
    - **replication_chance** - the tendency of forks in the road to occur (higher = denser road networks). A replication_chance of 0 may cause a generation fault. A very low value (on the order of 0.01) may also cause generation faults if the max_turn_speed is not also low (< 1.0).
    - **spontaneous_death_chance** - the tendency of spontaneous dead-ends to occur.
- **seed** - Integer seed for the internal np.random's random number generator. The same parameters with the same seed should yield identical networks. Use a seed of `None` for a randomly provided seed.


## Spatial Hashing
Grid partitioning is used to reduce the computational cost of running proximity checks on large numbers of lanes.
It vastly reduces the generation process time and is also used in `LaneLidarObservation` and `PartitionedRoadNetwork.get_closest_lane`

A set of functions that support grid partitioning is available under `generation/spatial_hash.py`
