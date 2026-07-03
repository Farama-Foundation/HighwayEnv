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
exit
lane_keeping
two_way
u_turn
```

HighwayEnv includes the following ten environments for autonomous driving decision-making. Each environment features configurable observations, actions, dynamics, and rewards — see {ref}`Configuring an environment <configuration>` for details.

| Environment | Description | Action type |
|---|---|---|
| {ref}`Highway <environments-highway>` | Drive fast on a multilane highway while avoiding collisions. | Discrete |
| {ref}`Merge <environments-merge>` | Merge onto a highway from an on-ramp through dense traffic. | Discrete |
| {ref}`Roundabout <environments-roundabout>` | Navigate a roundabout with merging and exiting traffic. | Discrete |
| {ref}`Parking <environments-parking>` | Park in a given space with the correct heading (goal-conditioned). | Continuous |
| {ref}`Intersection <environments-intersection>` | Cross an unsignalized intersection among other vehicles. | Discrete |
| {ref}`Racetrack <environments-racetrack>` | Follow a racetrack as fast as possible while staying on the road. | Continuous |
| {ref}`Exit <environments-exit>` | Navigate across lanes to reach a highway exit ramp. | Discrete |
| {ref}`Lane Keeping <environments-lane-keeping>` | Steer to follow a sine-wave lane using bicycle dynamics. | Continuous |
| {ref}`Two Way <environments-two-way>` | Overtake on a two-way road with oncoming traffic (risk management). | Discrete |
| {ref}`U-Turn <environments-u-turn>` | Overtake blocking vehicles through a double-lane U-turn. | Discrete |

All environments follow the [Gymnasium](https://gymnasium.farama.org/) API and are highly configurable via arguments specified in each environment's documentation.
