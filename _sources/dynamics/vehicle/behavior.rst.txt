.. _vehicle_behavior:

Behavior
==========

.. py:module::highway_env.vehicle.behavior

Other simulated vehicles follow simple and realistic behaviors that dictate how they accelerate and
steer on the road. They are implemented in the :py:class:`~highway_env.vehicle.behavior.IDMVehicle` class.

Longitudinal Behavior
~~~~~~~~~~~~~~~~~~~~~~

The acceleration of the vehicle is given by the *Intelligent Driver Model* (IDM) from :cite:`Treiber2000`.

.. math::
        \dot{v} &= a\left[1-\left(\frac{v}{v_0}\right)^\delta - \left(\frac{d^*}{d}\right)^2\right] \\
        d^* &= d_0 + Tv + \frac{v\Delta v}{2\sqrt{ab}} \\

where :math:`v` is the vehicle velocity, :math:`d` is the distance to its front vehicle.
The dynamics are parametrised by:

- :math:`v_0` the desired velocity, as :py:attr:`~highway_env.vehicle.behavior.IDMVehicle.target_velocity`
- :math:`T` the desired time gap, as :py:attr:`~highway_env.vehicle.behavior.IDMVehicle.TIME_WANTED`
- :math:`d_0` the jam distance, as :py:attr:`~highway_env.vehicle.behavior.IDMVehicle.DISTANCE_WANTED`
- :math:`\delta` the velocity exponent, as :py:attr:`~highway_env.vehicle.behavior.IDMVehicle.DELTA`

It is implemented in :py:meth:`~highway_env.vehicle.behavior.IDMVehicle.acceleration` method.

Lateral Behavior
~~~~~~~~~~~~~~~~

The discrete lane change decisions are given by the *Minimizing Overall Braking Induced by Lane change* (MOBIL) model from :cite:`Kesting2007`.
The model is parametrised by:

- :py:attr:`~highway_env.vehicle.behavior.IDMVehicle.POLITENESS`, a politeness coefficient
- :py:attr:`~highway_env.vehicle.behavior.IDMVehicle.LANE_CHANGE_MIN_ACC_GAIN`, the acceleration gain required to trigger a lane change
- :py:attr:`~highway_env.vehicle.behavior.IDMVehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED`, the maximum braking imposed to a vehicle in a cut-in
- :py:attr:`~highway_env.vehicle.behavior.IDMVehicle.LANE_CHANGE_DELAY`, the minimum delay between two lane changes


It is implemented in the :py:meth:`~highway_env.vehicle.behavior.IDMVehicle.mobil` method.

.. note::
    In the :py:class:`~highway_env.vehicle.behavior.LinearVehicle` class, the longitudinal and lateral behaviours
    are approximated as linear weightings of several features, such as the distance and speed difference to the leading
    vehicle.



API
***

.. automodule:: highway_env.vehicle.behavior
    :members:
    :inherited-members:
    :show-inheritance:
