(install)=

# Installation

## Prerequisites

### Optional tools

- [uv](https://docs.astral.sh/uv/) — fast Python package manager with lockfile support (recommended for contributors), install with [standalone installer](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) or simply `pip install uv`.
- [just](https://github.com/casey/just) — command runner used for common development tasks (see the `Justfile`), install [with your package manager of choice](https://github.com/casey/just#packages).

### Ubuntu

```bash
sudo apt-get update -y
sudo apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev \
    libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev \
    ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc
```

### macOS

We recommend using [Homebrew](https://brew.sh).
```bash
brew install uv    # optional
brew install just  # optional
```

### Windows 10

## Stable release

To install the latest stable version:

```bash
pip install highway-env
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install highway-env
```

## Development version

To install the current development version:

```bash
pip install --user git+https://github.com/Farama-Foundation/HighwayEnv
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install git+https://github.com/Farama-Foundation/HighwayEnv
```

## Contributing / Development setup

If you want to contribute or need reproducible dependency versions, see {ref}`faq-uv-frozen` for how to set up a development environment using uv with the pinned lockfile.
