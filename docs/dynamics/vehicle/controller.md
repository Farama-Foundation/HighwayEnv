(vehicle-controller)=

# Control

The {py:class}`~highway_env.vehicle.controller.ControlledVehicle` class implements a low-level controller on top of a {py:class}`~highway_env.vehicle.kinematics.Vehicle`, allowing to track a given target speed and follow a target lane.
The controls are computed when calling the {py:meth}`~highway_env.vehicle.controller.ControlledVehicle.act` method.

## Longitudinal controller

The longitudinal controller is a simple proportional controller:

$$
a = K_p(v_r - v),
$$

where

- $a$ is the vehicle acceleration (throttle);
- $v$ is the vehicle velocity;
- $v_r$ is the reference velocity;
- $K_p$ is the controller proportional gain, implemented as {py:attr}`~highway_env.vehicle.controller.ControlledVehicle.KP_A`.

It is implemented in the {py:meth}`~highway_env.vehicle.controller.ControlledVehicle.speed_control` method.

## Lateral controller

The lateral controller is a simple proportional-derivative controller, combined with some non-linearities that invert those of the {ref}`kinematics model <vehicle_kinematics>`.

### Position control

$$
v_{\text{lat},r} &= -K_{p,\text{lat}} \Delta_{\text{lat}}, \\ \Delta \psi_{r} &= \arcsin \left(\frac{v_{\text{lat},r}}{v}\right),
$$

### Heading control

$$
\psi_r &= \psi_L + \Delta \psi_{r}, \\ \dot{\psi}_r &= K_{p,\psi} (\psi_r - \psi), \\ \delta &= \arcsin \left(\frac{1}{2} \frac{l}{v} \dot{\psi}_r\right), \\
$$

where

- $\Delta_{\text{lat}}$ is the lateral position of the vehicle with respect to the lane center-line;
- $v_{\text{lat},r}$ is the lateral velocity command;
- $\Delta \psi_{r}$ is a heading variation to apply the lateral velocity command;
- $\psi_L$ is the lane heading (at some lookahead position to anticipate turns);
- $\psi_r$ is the target heading to follow the lane heading and position;
- $\dot{\psi}_r$ is the yaw rate command;
- $\delta$ is the front wheels angle control;
- $K_{p,\text{lat}}$ and $K_{p,\psi}$ are the position and heading control gains.

It is implemented in the {py:meth}`~highway_env.vehicle.controller.ControlledVehicle.steering_control` method.

## API

```{eval-rst}
.. automodule:: highway_env.vehicle.controller
    :members:
```
