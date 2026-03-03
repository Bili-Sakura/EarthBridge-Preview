#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Train CUT baseline for **sar2ir** (SAR → IR, 1-band → 1-band, 1024×1024).

Usage::

    # Single GPU
    python -m examples.cut.train_sar2ir

    # Multi-GPU via accelerate
    accelerate launch -m examples.cut.train_sar2ir --train_batch_size 4

All default hyper-parameters live in :func:`config.sar2ir_config`.
Any field can be overridden from the command line (``--field value``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import sar2ir_config, TaskConfig  # noqa: E402
from .trainer import CUTTrainer  # noqa: E402


def _bool_arg(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def _resolve_arg_type(field_name: str, field_val):
    if isinstance(field_val, bool):
        return _bool_arg
    if field_val is not None:
        return type(field_val)
    
    # Use get_type_hints to resolve string annotations (from __future__ import annotations)
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
    parser = argparse.ArgumentParser(description="Train CUT – sar2ir")
    for field_name, field_val in vars(sar2ir_config()).items():
        ftype = _resolve_arg_type(field_name, field_val)
        parser.add_argument(f"--{field_name}", type=ftype, default=field_val)
    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}


def main():
    overrides = parse_overrides()
    cfg = sar2ir_config(**overrides)

    # ------------------------------------------------------------------
    # Task-specific modifications can be added here.
    # For example:  cfg.ngf = 128
    # ------------------------------------------------------------------

    trainer = CUTTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
