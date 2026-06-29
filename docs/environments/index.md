(environments)=

# Environments

```{toctree}
:hidden:

highway
merge
roundabout
parking
intersection
racetrack
```

HighwayEnv includes the following six environments for autonomous driving decision-making. Each environment features configurable observations, actions, dynamics, and rewards — see {ref}`Configuring an environment <configuration>` for details.

| Environment | Description | Action type |
|---|---|---|
| {ref}`Highway <environments-highway>` | Drive fast on a multilane highway while avoiding collisions. | Discrete |
| {ref}`Merge <environments-merge>` | Merge onto a highway from an on-ramp through dense traffic. | Discrete |
| {ref}`Roundabout <environments-roundabout>` | Navigate a roundabout with merging and exiting traffic. | Discrete |
| {ref}`Parking <environments-parking>` | Park in a given space with the correct heading (goal-conditioned). | Continuous |
| {ref}`Intersection <environments-intersection>` | Cross an unsignalized intersection among other vehicles. | Discrete |
| {ref}`Racetrack <environments-racetrack>` | Follow a racetrack as fast as possible while staying on the road. | Continuous |

All environments follow the [Gymnasium](https://gymnasium.farama.org/) API and are highly configurable via arguments specified in each environment's documentation.
