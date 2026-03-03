#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Count model parameters for DBIM training configs."""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.unet_dbim import create_dbim_model


def count_model(num_channels, num_res_blocks, attention_resolutions, channel_mult):
    model = create_dbim_model(
        image_size=512,
        in_channels=1,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        unet_type="adm",
        attention_resolutions=attention_resolutions,
        dropout=0.0,
        condition_mode="concat",
        channel_mult=channel_mult,
        attention_head_dim=64,
    )
    return sum(p.numel() for p in model.parameters())


ARCH_SEARCH_CONFIGS = [
    ("cuda0_4st_noattn", 64, 2, "", "1,2,3,4"),
    ("cuda1_5st_nc96", 96, 2, "", "1,2,4,4,8"),
    ("cuda2_6st_noattn", 64, 2, "", "1,1,2,2,4,4"),
    ("cuda3_attn32", 64, 2, "32", "1,2,4,4,8"),
    ("cuda4_nrb3", 64, 3, "", "1,2,4,4,8"),
    ("cuda5_nc64", 64, 2, "", "1,2,4,4,8"),  # v3 baseline
    ("cuda6_nc128", 128, 2, "", "1,2,3,4"),
    ("cuda7_nc32", 32, 2, "", "1,2,4,4,8"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch-search", action="store_true", help="Count all 8 arch-search configs")
    parser.add_argument("--config", type=str, help="Single config name (e.g. cuda0_4st_noattn)")
    args = parser.parse_args()

    if args.arch_search:
        for name, nc, nrb, attn, cm in ARCH_SEARCH_CONFIGS:
            n = count_model(nc, nrb, attn, cm)
            print(f"{name}: {n:,} ({n/1e6:.2f}M)")
        return

    if args.config:
        for name, nc, nrb, attn, cm in ARCH_SEARCH_CONFIGS:
            if name == args.config:
                n = count_model(nc, nrb, attn, cm)
                print(n)
                return
        print(f"Unknown config: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Default: v3 config
    n = count_model(64, 2, "", "1,2,4,4,8")
    print(f"Total parameters: {n:,} ({n/1e6:.2f}M)")


if __name__ == "__main__":
    main()
