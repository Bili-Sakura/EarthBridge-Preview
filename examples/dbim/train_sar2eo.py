#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Train DBIM baseline for sar2eo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import sar2eo_config, TaskConfig  # noqa: E402
from .trainer import DBIMTrainer  # noqa: E402


def _bool_arg(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def _resolve_arg_type(field_name: str, field_val):
    if isinstance(field_val, bool):
        return _bool_arg
    if field_val is not None:
        return type(field_val)
    hints = get_type_hints(TaskConfig)
    annotation = hints.get(field_name, str)
    origin = get_origin(annotation)
    if origin is Union:
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]
    if annotation is bool:
        return _bool_arg
    if annotation in (int, float, str):
        return annotation
    return str


def parse_overrides() -> dict:
    parser = argparse.ArgumentParser(description="Train DBIM – sar2eo")
    for field_name, field_val in vars(sar2eo_config()).items():
        parser.add_argument(f"--{field_name}", type=_resolve_arg_type(field_name, field_val), default=field_val)
    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}


def main():
    cfg = sar2eo_config(**parse_overrides())
    trainer = DBIMTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
