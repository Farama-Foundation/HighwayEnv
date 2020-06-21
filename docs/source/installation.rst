.. _install:

Installation
============

Prerequisites
-------------

This project requires python3 (>=3.5)

The graphics require the installation of `pygame <https://www.pygame.org/news>`_, which itself has dependencies that must be installed manually.


Ubuntu
~~~~~~

.. code-block:: bash

    sudo apt-get update -y
    sudo apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev
        libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev
        ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc

Windows 10
~~~~~~~~~~

We recommend using `Anaconda <https://conda.io/docs/user-guide/install/windows.html>`_.


Latest release
--------------
To install the latest master version locally:

.. code-block:: bash

    pip install --user git+https://github.com/eleurent/highway-env
