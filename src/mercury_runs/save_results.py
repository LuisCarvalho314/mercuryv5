# src/mercury_runs/save_results.py
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as import_error:  # pragma: no cover
    raise ImportError("pyarrow is required for parquet metadata support. Install with: pip install pyarrow") from import_error

from .schemas_results import ResultBundleMeta


BUNDLE_META_KEY = "mercury:result_bundle_meta_json"
BUNDLE_SCHEMA_VERSION_KEY = "mercury:bundle_schema_version"
BUNDLE_SCHEMA_VERSION = "1"


def _encode_kv_metadata(mapping: Dict[str, str]) -> Dict[bytes, bytes]:
    return {key.encode("utf-8"): value.encode("utf-8") for key, value in mapping.items()}


def write_bundle_parquet(
    *,
    output_dir: Path,
    run_id: str,
    bundle_name: str,
    columns: Dict[str, Any],
    meta: ResultBundleMeta,
    embed_metadata_in_parquet: bool = True,
    parquet_compression: str = "zstd",
    parquet_statistics: bool = True,
) -> Path:
    """
    Writes a parquet file named: <run_id>_<bundle_name>.parquet

    If embed_metadata_in_parquet=True, embeds ResultBundleMeta as key-value parquet metadata
    so you can later retrieve parameters without external JSON files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{run_id}_{bundle_name}.parquet"
    tmp_path = parquet_path.with_name(f".{parquet_path.name}.{uuid.uuid4().hex}.tmp")

    data_frame = pl.DataFrame(columns)

    if not embed_metadata_in_parquet:
        data_frame.write_parquet(tmp_path, compression=parquet_compression, statistics=parquet_statistics)
        os.replace(tmp_path, parquet_path)
        return parquet_path

    # Convert to Arrow Table for reliable metadata embedding
    arrow_table = data_frame.to_arrow()

    meta_json = meta.model_dump_json()  # pydantic v2; for v1 use meta.json()
    kv_metadata = {
        BUNDLE_META_KEY: meta_json,
        BUNDLE_SCHEMA_VERSION_KEY: BUNDLE_SCHEMA_VERSION,
    }

    existing_schema_metadata = arrow_table.schema.metadata or {}
    merged_metadata: Dict[bytes, bytes] = dict(existing_schema_metadata)
    merged_metadata.update(_encode_kv_metadata(kv_metadata))

    arrow_table_with_metadata = arrow_table.replace_schema_metadata(merged_metadata)

    pq.write_table(
        arrow_table_with_metadata,
        tmp_path,
        compression=parquet_compression,
        write_statistics=parquet_statistics,
    )
    os.replace(tmp_path, parquet_path)

    return parquet_path


def read_bundle_meta(parquet_path: Path) -> Optional[ResultBundleMeta]:
    """
    Reads ResultBundleMeta embedded in parquet key-value metadata.
    Returns None if not present.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    file_metadata = parquet_file.metadata
    if file_metadata is None or file_metadata.metadata is None:
        return None

    kv = file_metadata.metadata  # Dict[bytes, bytes]
    raw = kv.get(BUNDLE_META_KEY.encode("utf-8"))
    if raw is None:
        return None

    meta_dict = json.loads(raw.decode("utf-8"))
    return ResultBundleMeta.model_validate(meta_dict)  # pydantic v2; for v1 use ResultBundleMeta.parse_obj(...)
