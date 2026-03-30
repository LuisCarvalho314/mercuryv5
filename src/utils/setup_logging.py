import os
import json
import logging
import logging.config
from pathlib import Path


def setup_logging(
    default_path=Path(__file__).parent / "config.json",
    default_level=logging.INFO,
    env_key="LOG_CFG",
):
    """Setup logging configuration."""
    path = default_path
    value = os.getenv(env_key)
    if value:
        path = value

    path = Path(path)

    if path.exists():
        with path.open("rt") as file_handle:
            config = json.load(file_handle)

        # Ensure directories exist for any handlers that write to files
        handlers = config.get("handlers", {})
        for handler_config in handlers.values():
            filename = handler_config.get("filename")
            if filename:
                Path(filename).expanduser().resolve().parent.mkdir(
                    parents=True, exist_ok=True
                )

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
