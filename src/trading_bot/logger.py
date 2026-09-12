"""Configuration centralisée du logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = "logs/trading_bot.log") -> logging.Logger:
    logger = logging.getLogger("trading_bot")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("trading_bot")
