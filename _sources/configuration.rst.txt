.. _configuration:

Configuring an environment
==========================

The observation, action, dynamics and rewards of an environment are all specified in its configuration, defined as a
dictionary :py:attr:`~highway_env.envs.common.abstract.AbstractEnv.config`.

After environment creation, its configuration can be changed using the
:py:meth:`~highway_env.envs.common.abstract.AbstractEnv.configure` method.