# Architecture

The movable codebase is now organized as a layered application under `src/mercury_runs`.

## Layers

- `domain/`
  - Typed Pydantic models and contracts for runs, studies, artifacts, and statuses.
- `algorithms/`
  - Canonical Python-first algorithm packages for Mercury, POCML, and CSCG, each split into step modules.
- `application/`
  - Single-run and study use cases.
- `infrastructure/`
  - Filesystem persistence, artifact layout, reporting, W&B logging, subprocess execution, and runtime helpers.
- `interfaces/`
  - CLI parsing and top-level dispatch.

## Protected Dependencies

These directories are treated as external dependencies and are not modified by the rebuild:

- `external/`
- `src/mercury/`
- `src/maze_environment/`

All interaction with those directories should happen through adapters in the movable layers rather than direct business logic in the CLI.

## Canonical Surface

The canonical implementation now lives in:

- `algorithms/`
- `application/`
- `domain/`
- `infrastructure/`
- `interfaces/`

The old top-level wrapper modules were removed. New code should import canonical modules directly.
