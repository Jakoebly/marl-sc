#!/usr/bin/env python3
"""
Standalone script to test environment config loading (including synthetic data
and lead_time_sampler replacement). Loads config as in experiments but does
not start a run; only prints the loaded config.

Run from project root:
    python tests/test_env_config_loading.py [config_path]

If no config_path is given, uses config_files/environments/base_env.yaml.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Load environment config (as in experiments) and print it."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(project_root / "config_files" / "environments" / "base_env.yaml"),
        help="Path to environment config YAML (default: config_files/environments/base_env.yaml)",
    )
    args = parser.parse_args()

    from src.config.loader import load_environment_config
    from src.utils.seed_manager import SeedManager, EXPERIMENT_SEEDS

    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 60)
    print("Environment Config Loading Test")
    print("=" * 60)
    print(f"Config path: {config_path}")
    print()

    seed_manager = SeedManager(root_seed=42, seed_registry=EXPERIMENT_SEEDS)
    env_config = load_environment_config(str(config_path), seed_manager=seed_manager)

    # Convert to dict for readable printing (Pydantic model_dump)
    config_dict = env_config.model_dump(mode="json")

    print("Loaded config (relevant sections):")
    print("-" * 60)


    # Verify replacement worked for synthetic
    lts = config_dict["components"]["lead_time_sampler"]
    if lts.get("type") == "custom" and "values" in lts.get("params", {}):
        vals = lts["params"]["values"]
        nw, ns = config_dict["n_warehouses"], config_dict["n_skus"]
        if len(vals) == nw and all(len(row) == ns for row in vals):
            print(f"[OK] lead_time_sampler replaced with custom values, shape ({nw}, {ns})")
        else:
            print(
                f"[WARN] lead_time_sampler has custom values but shape mismatch: "
                f"got ({len(vals)}, {len(vals[0]) if vals else 0}), expected ({nw}, {ns})"
            )
    else:
        print("[INFO] lead_time_sampler not custom (data_source may be real_world)")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
