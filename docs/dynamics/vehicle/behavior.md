(vehicle-behavior)=

# Behavior

% py:module::highway_env.vehicle.behavior

Other simulated vehicles follow simple and realistic behaviors that dictate how they accelerate and
steer on the road. They are implemented in the {py:class}`~highway_env.vehicle.behavior.IDMVehicle` class.

## Longitudinal Behavior

The acceleration of the vehicle is given by the *Intelligent Driver Model* (IDM) from {cite}`Treiber2000`.

$$
\dot{v} &= a\left[1-\left(\frac{v}{v_0}\right)^\delta - \left(\frac{d^*}{d}\right)^2\right] \\ d^* &= d_0 + Tv + \frac{v\Delta v}{2\sqrt{ab}} \\
$$

where $v$ is the vehicle velocity, $d$ is the distance to its front vehicle.
The dynamics are parametrised by:

- $v_0$ the desired velocity, as {py:attr}`~highway_env.vehicle.behavior.IDMVehicle.target_velocity`
- $T$ the desired time gap, as {py:attr}`~highway_env.vehicle.behavior.IDMVehicle.TIME_WANTED`
- $d_0$ the jam distance, as {py:attr}`~highway_env.vehicle.behavior.IDMVehicle.DISTANCE_WANTED`
- $a,\,b$ the maximum acceleration and deceleration, as {py:attr}`~highway_env.vehicle.behavior.IDMVehicle.COMFORT_ACC_MAX` and {py:attr}`~highway_env.vehicle.behavior.IDMVehicle.COMFORT_ACC_MIN`
- $\delta$ the velocity exponent, as {py:attr}`~highway_env.vehicle.behavior.IDMVehicle.DELTA`

It is implemented in {py:meth}`~highway_env.vehicle.behavior.IDMVehicle.acceleration` method.

## Lateral Behavior

The discrete lane change decisions are given by the *Minimizing Overall Braking Induced by Lane change* (MOBIL) model from {cite}`Kesting2007`.
According to this model, a vehicle decides to change lane when:

- it is **safe** (do not cut-in):

$$
\tilde{a}_n \geq - b_\text{safe};
$$

- there is an **incentive** (for the ego-vehicle and possibly its followers):

$$
\underbrace{\tilde{a}_c - a_c}_{\text{ego-vehicle}} + p\left(\underbrace{\tilde{a}_n - a_n}_{\text{new follower}} + \underbrace{\tilde{a}_o - a_o}_{\text{old follower}}\right) \geq \Delta a_\text{th},
$$

where

- $c$ is the center (ego-) vehicle, $o$ is its old follower *before* the lane change, and $n$ is its new follower *after* the lane change
- $a, \tilde{a}$ are the acceleration of the vehicles *before* and *after* the lane change, respectively.
- $p$ is a politeness coefficient, implemented as {py:attr}`~highway_env.vehicle.behavior.IDMVehicle.POLITENESS`
- $\Delta a_\text{th}$ the acceleration gain required to trigger a lane change, implemented as {py:attr}`~highway_env.vehicle.behavior.IDMVehicle.LANE_CHANGE_MIN_ACC_GAIN`
- $b_\text{safe}$ the maximum braking imposed to a vehicle during a cut-in, implemented as {py:attr}`~highway_env.vehicle.behavior.IDMVehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED`

It is implemented in the {py:meth}`~highway_env.vehicle.behavior.IDMVehicle.mobil` method.

```{note}
In the {py:class}`~highway_env.vehicle.behavior.LinearVehicle` class, the longitudinal and lateral behaviours
are approximated as linear weightings of several features, such as the distance and speed difference to the leading
vehicle.
```

### API

```{eval-rst}
.. automodule:: highway_env.vehicle.behavior
    :members:
```
