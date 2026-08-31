from highway_env import __version__


class HighwayEnvExperimentalWarning(FutureWarning):
    """The environment is still experimental and not stable yet."""

    template: str = (
        "\033[31mhighway_env.envs:\033[0m The environment [%s] is not yet stable in "
        f"current version of HighwayEnv ({__version__}), expect breaking change "
        "in behaviour or API design."
    )
