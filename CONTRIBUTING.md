# HighwayEnv Contribution Guidelines

We welcome:

- Bug reports
- Pull requests for bug fixes
- Documentation improvements
- New environments (see [Make your own environment](https://farama-foundation.github.io/HighwayEnv/make_your_own/) for the process; open an issue first if it's a significant addition so we can align on scope)
- Features and enhancements to existing environments

Notably, changes to existing environment behavior (reward shaping, observation/action spaces, dynamics) should be minimized where possible. RL results are sensitive to environment version, so any behavior change requires a version bump and a changelog entry, and should be flagged clearly in the PR description.

## Contributing to the codebase

### Coding

Contributing code is done through standard GitHub methods:

1. Fork this repo
2. Create a branch and commit your code
3. Submit a pull request. A maintainer will review and give feedback or request changes as needed

### Considerations

- Install with `pip install -e .` and make sure existing tests pass with `pytest -v` before opening a PR
- New code should be tested; if you're fixing a bug, add a regression test that fails without your fix
- Any environment behavior fix should include corresponding documentation updates
- Keep PRs focused. Large PRs mixing unrelated changes are harder to review and slower to merge

### Git hooks

The CI runs several checks on new code. You can run these locally instead of waiting on CI:

1. [Install `pre-commit`](https://pre-commit.com/#install)
2. Install the Git hooks: `pre-commit install`

Once installed, hooks run automatically on every commit. You can also run them manually with `pre-commit run --all-files`, or skip them (not recommended) with `git commit --no-verify`.

**Note:** the first `pre-commit run --all-files` may fail after auto-formatting files; run it a second time and it should pass.
