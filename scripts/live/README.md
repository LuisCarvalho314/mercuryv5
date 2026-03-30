# Live Scripts

The legacy live/debug scripts were removed during the rebuild.

If interactive tooling is needed again, it should be rebuilt on top of:
- `src/mercury_runs/algorithms/*` for fine-grained algorithm control
- `src/mercury_runs/application/*` for run/study orchestration
- `src/mercury_runs/interfaces/cli.py` for CLI integration

Do not restore the old one-off wrapper scripts.
