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

### Test coverage

CI enforces unit-test coverage on pull requests and pushes to `main` and branches under `test/` (for example `test/my-feature`).

| Check | Threshold | Scope | Workflow |
|-------|-----------|-------|----------|
| **Total coverage** | ≥ 85% | `highway_env` package | `coverage (total)` |
| **Diff coverage** | ≥ 80% | Lines changed in `highway_env` | `coverage (diff)` |

Diff coverage only applies to lines you add or modify in `highway_env/`; it does not require every legacy gap in the package to be filled in one PR. Total coverage guards the overall baseline.

Run the same checks locally before opening a PR:

```bash
just coverage                              # both checks (diff vs origin/main)
just coverage-total                        # ≥ 85% on highway_env
just coverage-diff                         # ≥ 80% on changed highway_env lines
just coverage-diff upstream                # diff vs upstream/main
just coverage-diff upstream my-feature     # diff vs upstream/my-feature
```

Pass `remote` and `branch` as positional arguments (`just coverage-diff upstream my-feature`).

New or changed behaviour in `highway_env/` should include tests that exercise it. If you are fixing a bug, add a regression test that fails without your fix.

### Fork & merge

Contributing code is done through the same development process like most GitHub-based projects:

1. Fork this repo
2. Create a branch and commit your code
3. Submit a pull request, a maintainer will review and give feedback or request changes as needed, they may also commit to your fork directly for certain changes

### Considerations

- Install developer dependencies with `just install` and make sure existing tests pass with `just test` before opening a PR
- Meet the [test coverage requirements](#test-coverage): ≥ 85% total on `highway_env`, ≥ 80% on lines changed in `highway_env`
- Any user-facing environment behaviour change should include corresponding documentation updates
- Keep PRs focused, large PRs mixing unrelated changes are harder to review and slower to merge (break it up!)
