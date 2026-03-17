# scCChain Branding Naming Design

**Date:** 2026-03-17

**Goal:** Standardize the human-facing project name to `scCChain` across the repository while preserving `ScCChain` anywhere it is required as the Julia package, module, or repository identifier.

## Scope

- Rewrite prose occurrences of `ScCChain` to `scCChain` in documentation, comments, docstrings, and notebook narrative text.
- Preserve `ScCChain` in code identifiers and commands, including `module ScCChain`, `using ScCChain`, `Pkg.add(...)`, file names, and repository URLs.
- Preserve the repository/package display name `ScCChain` where it explicitly names the Julia package or repository.

## Decision

Use a conservative text-only cleanup:

- Keep package and repository identifiers untouched.
- Change only narrative references that describe the project in plain language.
- Verify with targeted searches so remaining `ScCChain` usages are only the intentional identifier/repository cases.

## Risks

- Notebook JSON can mix prose and executable code, so replacements must be limited to exact narrative strings.
- README and script headers contain both prose and code examples; only the prose lines should change.

## Non-Goals

- No package/module rename.
- No repository rename.
- No commits in this pass.
