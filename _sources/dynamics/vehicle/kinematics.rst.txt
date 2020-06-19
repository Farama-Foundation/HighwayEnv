.. _vehicle_kinematics:

.. py:module::highway_env.vehicle.kinematics

Kinematics
==================

The vehicles kinematics are represented in the :py:class:`~highway_env.vehicle.kinematics.Vehicle` class by the *Kinematic Bicycle Model* :cite:`Polack2017`.

.. math::
        \dot{x}&=v\cos(\psi+\beta) \\
        \dot{y}&=v\sin(\psi+\beta) \\
        \dot{v}&=a \\
        \dot{\psi}&=\frac{v}{l}\sin\beta \\
        \beta&=\tan^{-1}(1/2\tan\delta), \\

where

- :math:`(x, y)` is the vehicle position;
- :math:`v` its forward speed;
- :math:`\psi` its heading;
- :math:`a` is the acceleration command;
- :math:`\beta` is the slip angle at the center of gravity;
- :math:`\delta` is the front wheel angle used as a steering command.

These calculations appear in the :py:meth:`~highway_env.vehicle.kinematics.Vehicle.step` method.

API
***

.. automodule:: highway_env.vehicle.kinematics
    :members:

