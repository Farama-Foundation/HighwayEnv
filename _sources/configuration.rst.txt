.. _configuration:

Configuring an environment
==========================

The :ref:`observations <observations>`, :ref:`actions <actions>`, :ref:`dynamics <dynamics>` and :ref:`rewards <rewards>`
of an environment are parametrized by a configuration, defined as a
:py:attr:`~highway_env.envs.common.abstract.AbstractEnv.config` dictionary.

After environment creation, its configuration can be changed using the
:py:meth:`~highway_env.envs.common.abstract.AbstractEnv.configure` method.