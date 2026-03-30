from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .runtime import atomic_write_json, load_json, utc_now


def progress_path(run_root: Path) -> Path:
    return run_root / "progress.json"


def write_progress(
    *,
    run_root: Path,
    run_id: str,
    method: str,
    track: str = "lifecycle",
    stage: str,
    current: Optional[int] = None,
    total: Optional[int] = None,
    message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = progress_path(run_root)
    existing = load_json(path) or {}
    tracks = dict(existing.get("tracks") or {})
    tracks[track] = {
        "track": track,
        "stage": stage,
        "current": current,
        "total": total,
        "message": message,
        "extra": extra or {},
        "updated_at_utc": utc_now(),
    }
    payload: Dict[str, Any] = {"run_id": run_id, "method": method, "tracks": tracks, "updated_at_utc": utc_now()}
    atomic_write_json(path, payload)
    return path


def load_progress(run_root: Path) -> Optional[Dict[str, Any]]:
    return load_json(progress_path(run_root))


def load_progress_tracks(run_root: Path) -> Dict[str, Dict[str, Any]]:
    progress = load_progress(run_root) or {}
    tracks = progress.get("tracks")
    if isinstance(tracks, dict):
        return {str(name): dict(value) for name, value in tracks.items() if isinstance(value, dict)}
    if progress:
        return {
            "lifecycle": {
                "track": "lifecycle",
                "stage": progress.get("stage"),
                "current": progress.get("current"),
                "total": progress.get("total"),
                "message": progress.get("message"),
                "extra": progress.get("extra") or {},
                "updated_at_utc": progress.get("updated_at_utc"),
            }
        }
    return {}
