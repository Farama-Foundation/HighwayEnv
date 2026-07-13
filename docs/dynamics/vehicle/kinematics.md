(vehicle_kinematics)=

% py:module::highway_env.vehicle.kinematics

# Kinematics

The vehicles kinematics are represented in the {py:class}`~highway_env.vehicle.kinematics.Vehicle` class by the *Kinematic Bicycle Model* {cite}`Polack2017`.

$$
\dot{x}&=v\cos(\psi+\beta) \\ \dot{y}&=v\sin(\psi+\beta) \\ \dot{v}&=a \\ \dot{\psi}&=\frac{v}{l}\sin\beta \\ \beta&=\tan^{-1}(1/2\tan\delta), \\
$$

where

- $(x, y)$ is the vehicle position;
- $v$ its forward speed;
- $\psi$ its heading;
- $a$ is the acceleration command;
- $\beta$ is the slip angle at the center of gravity;
- $\delta$ is the front wheel angle used as a steering command.

These calculations appear in the {py:meth}`~highway_env.vehicle.kinematics.Vehicle.step` method.

## API

```{eval-rst}
.. automodule:: highway_env.vehicle.kinematics
    :members:
```
