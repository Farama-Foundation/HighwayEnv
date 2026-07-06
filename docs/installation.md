(install)=

# Installation

To install the latest stable version, simply install with `pip`:

```bash
pip install highway-env
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install highway-env
```

## Special cases

### Ubuntu

If you are on Linux, you may need additional dependencies for [pygame-ce](https://pyga.me), which provides the graphics:

```bash
sudo apt-get update -y
sudo apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev \
    libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev \
    ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc
```

### Development version

If you want to install the current development version (unstable!):

```bash
pip install git+https://github.com/Farama-Foundation/HighwayEnv
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install git+https://github.com/Farama-Foundation/HighwayEnv
```

### Contributing / Development setup

If you want to contribute to HighwayEnv (insert smiley face here), these tools would be helpful to developers:
- [uv](https://docs.astral.sh/uv/) — fast Python package manager with lockfile support (recommended for contributors), install with [standalone installer](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) or simply `pip install uv`.
- [just](https://github.com/casey/just) — command runner used for common development tasks (see the `Justfile`), install [with your package manager of choice](https://github.com/casey/just/blob/master/README.md#packages).

Check {ref}`faq-uv-frozen` for how to set up a development environment using uv with the pinned lockfile, and {ref}`faq-coverage` for the unit test coverage requirements enforced in CI.

See the full [contribution guidelines](https://github.com/Farama-Foundation/HighwayEnv/blob/main/CONTRIBUTING.md) on GitHub.
