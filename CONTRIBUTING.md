# HighwayEnv Contribution Guidelines

We welcome:

- Bug reports & feature proposal (you don't necessarily have to implement yourself)
- Pull requests for bug fixes & enhancements
- Documentation improvements
- New environments (see [Make your own environment](https://farama-foundation.github.io/HighwayEnv/main/make_your_own/) for the process; open an issue first if it's a significant addition so we can align on the scope & design choices)
- Features and enhancements to existing environments

For the sake of reproducibility, changes to existing environment behaviour (reward shaping, observation/action spaces, dynamics) should be avoided where possible. RL results are quite sensitive to environment variations, so any behaviour change may require a version bump and brief explanation in documentation, these should be flagged clearly in the PR description.

## Contributing to the codebase

### Developer tools

The recommended development workflow uses [`uv`](https://docs.astral.sh/uv/) and [`just`](https://github.com/casey/just).

Common commands:

- Install development dependencies: `just install`
- Run tests: `just test`
- Build documentation: `just docs-build`

Note that `just install` will also set up [`pre-commit`](https://pre-commit.com/). After installed, the configured pre-commit hooks will run automatically on every commit to act as a sanity check. You can also run them manually with `uv run --frozen pre-commit run --all-files`, or skip them (not recommended) with `git commit --no-verify`.

### Fork & merge

Contributing code is done through the same development process like most GitHub-based projects:

1. Fork this repo
2. Create a branch and commit your code
3. Submit a pull request, a maintainer will review and give feedback or request changes as needed, they may also commit to your fork directly for certain changes

### Considerations

- Install developer dependencies with `just install` and make sure existing tests pass with `just test` before opening a PR
- New code should be tested; if you're fixing a bug, add a regression test that fails without your fix
- Any user-facing environment behaviour change should include corresponding documentation updates
- Keep PRs focused, large PRs mixing unrelated changes are harder to review and slower to merge (break it up!)
