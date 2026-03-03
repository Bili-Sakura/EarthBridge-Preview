# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""DBIM dataset wrappers.

DBIM uses the same paired dataset structure as DDBM.
"""

from examples.ddbm.dataset_wrapper import (  # noqa: F401
    MavicTDDBMDataset as MavicTDBIMDataset,
    PairedValDataset,
)

__all__ = ["MavicTDBIMDataset", "PairedValDataset"]
