# RandomRoadEnv
**NOTE**: The following describes the environment as having multiple agents. Currently, RandomRoadEnv only supports a single agent. Multi-agent support is to be added soon.


Agent has low-level controls of a car and needs to navigate a procedurally generated road network to a goal parking spot.
There can be many agents in a small area, potentially creating dense traffic scenarios. Although no two agents will be assigned the same parking spot, their paths will likely intersect.

The rules are minimal:
1. Get to your destination in as little time as possible
2. Don't crash


## Observation
Tuple of 4 separate observation-types:
- NavigationObservation (directs agent as to what turn to make at the next junction)
- LidarObservation (observation of other vehicles)
- LaneLidarObservation (observation of road borders)
- RelativeGoalObservation (observation of relative position and alignment with goal spot)


## Rewards
- Curb collision penalty
- Vehicle-Vehicle collision penalty (worse than curb collision)
- Parking reward
    - Once a threshold of distance + alignment + speed is reached, the agent is considered parked and a one-time reward spike is given followed by termination
	- Recommended to decay the threshold from a higher, more lax value to a smaller, stricter value
- Timestep punishment (incentivizes speed)
- Route-following reward (scalar projection of velocity onto navigation arrow)
	- Crucial in the initial stages of learning but should be eventually decayed to 0 so that strategically moving backwards is not penalized


## Termination & Truncation
Termination occurs when an agent either crashes or parks successfully.
Truncation will occur after a fixed number of timesteps, or when all vehicles get terminated.


## Motivation
The goal is to have agents that have a generalized set of driving skills to any road network and are able to selfishly negotiate with other agents at intersections.

## Possible additional environment features
- Varying car specs (naturally slower and faster cars to encourage overtaking)
- Varying car sizes

## Todo
- Implement support for multiple agents
