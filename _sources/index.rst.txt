.. highway-env documentation master file, created by
   sphinx-quickstart on Wed Feb 28 15:51:44 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. |Build Status| image:: https://github.com/eleurent/highway-env/workflows/build/badge.svg
   :target: https://github.com/eleurent/highway-env/workflows/build/

.. |Documentation Status| image:: https://readthedocs.org/projects/highway-env/badge/?version=latest
   :target: https://highway-env.readthedocs.io/en/latest/

.. |Coverage Status| image:: https://codecov.io/gh/eleurent/highway-env/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/eleurent/highway-env

.. |Contributors| image:: https://img.shields.io/github/contributors/eleurent/highway-env
  :target: https://github.com/eleurent/highway-env/graphs/contributors


|Build Status| |Documentation Status| |Coverage Status| |Contributors|

Welcome to `highway-env <https://github.com/eleurent/highway-env>`_'s documentation!
====================================================================================

This project gathers a collection of environment for decision-making in Autonomous Driving. In particular, it focuses on scenarios of **interactions** with neighbour vehicles.

The purpose of this documentation is to provide:

- a description of the environments, and how they can be configured
- instructions on how to contribute and add your own environments


.. _index_installation_instructions:

Installation instructions
=========================

See the `installation instructions <https://github.com/eleurent/highway-env#installation>`_ on GitHub.

.. _index_how_to_cite_this_work:

How to cite this work?
======================

If you use this package, please consider citing it with this piece of
BibTeX:

.. code:: bibtex

  @misc{highway-env,
    author = {Leurent, Edouard},
    title = {An Environment for Autonomous Driving Decision-Making},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/eleurent/highway-env}},
  }

Documentation contents
======================

.. toctree::
  :maxdepth: 2

  environments/index
  observations/index
  actions/index
  dynamics/index
  rewards/index
  bibliography/index
