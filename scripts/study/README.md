# Study Scripts

Study execution is now owned by `main.py` and the `make study*` targets.

This folder is intentionally retired. Use:

```bash
uv run python main.py --study --study_config configs/studies/default.json
```

Or the supported convenience targets:

```bash
uv run make study STUDY_NAME=default MAX_WORKERS=4
uv run make study-resume STUDY_NAME=default MAX_WORKERS=4
uv run make study-retry STUDY_NAME=default MAX_WORKERS=4
```
