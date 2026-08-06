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
lane_keeping
two_way
exit
u_turn
random_road
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
| {ref}`Lane Keeping <environments-lane-keeping>` | Steer to follow a sine-wave lane using bicycle dynamics. | Continuous |
| {ref}`Two Way <environments-two-way>` | Overtake on a two-way road with oncoming traffic (risk management). | Discrete |
| {ref}`Exit <environments-exit>` | Navigate across lanes to reach a highway exit ramp. | Discrete |
| {ref}`U-Turn <environments-u-turn>` | Overtake blocking vehicles through a double-lane U-turn. | Discrete |
| {ref}`Random Road <environments-random-road>` | Navigate a procedurally generated road network to a goal parking spot. | Continuous |

All environments follow the [Gymnasium](https://gymnasium.farama.org/) API and are highly configurable via arguments specified in each environment's documentation.
