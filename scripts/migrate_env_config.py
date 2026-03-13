"""
Migrate env_config.yaml files from old action space schema to new.

Old format: max_order_quantities at top level (scalar or list).
New format: action_space.type + action_space.params.max_order_quantities.

Usage:
  python scripts/migrate_env_configs.py <ParentFolder>
  python scripts/migrate_env_configs.py WorkingConfig_Phase1.2
  python scripts/migrate_env_configs.py experiment_outputs/WorkingConfig_Phase1.2

Loops over each subfolder under the parent, opens env_config.yaml if present,
applies the migration, and saves the file. Use --dry-run to preview without writing.
"""

import argparse
import sys
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import yaml


def _needs_migration(config_dict: dict) -> bool:
    """Check if config uses old schema (max_order_quantities at top level, no action_space)."""
    if "action_space" in config_dict and config_dict["action_space"] is not None:
        action_space = config_dict["action_space"]
        if isinstance(action_space, dict):
            if action_space.get("type") and "params" in action_space:
                return False  # Already new format
    return "max_order_quantities" in config_dict


def _migrate_config(config_dict: dict) -> dict:
    """Migrate old format to new action_space schema. Modifies config_dict in place."""
    if not _needs_migration(config_dict):
        return config_dict

    max_qty = config_dict.pop("max_order_quantities", None)
    if max_qty is None:
        return config_dict

    n_skus = config_dict.get("n_skus", 1)
    if isinstance(max_qty, (int, float)):
        max_order_quantities = [int(max_qty)] * n_skus
    else:
        max_order_quantities = [int(x) for x in max_qty]

    config_dict["action_space"] = {
        "type": "direct",
        "params": {"max_order_quantities": max_order_quantities},
    }
    return config_dict


def migrate_env_config_file(path: Path, dry_run: bool = False) -> bool:
    """
    Migrate a single env_config.yaml file. Returns True if migration was applied.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return False

    # Handle optional "environment" wrapper (saved by base.py)
    if "environment" in data:
        config_dict = data["environment"]
        has_wrapper = True
    else:
        config_dict = data
        has_wrapper = False

    if not _needs_migration(config_dict):
        return False

    _migrate_config(config_dict)

    if has_wrapper:
        data["environment"] = config_dict
    else:
        data = config_dict

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate env_config.yaml files from old to new action space schema."
    )
    parser.add_argument(
        "parent_dir",
        type=str,
        help="Folder containing experiment subfolders (e.g. WorkingConfig_Phase1.2 or experiment_outputs/WorkingConfig_Phase1.2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    args = parser.parse_args()

    parent = Path(args.parent_dir)
    if not parent.is_dir():
        # Try under experiment_outputs (like run_evaluation_batch.sh)
        alt = project_root / "experiment_outputs" / args.parent_dir
        if alt.is_dir():
            parent = alt
            print(f"Resolved to: {parent}")
        else:
            print(f"[ERROR] Directory not found: {args.parent_dir}")
            sys.exit(1)

    migrated = 0
    skipped = 0
    no_config = 0

    for subdir in sorted(parent.iterdir()):
        if not subdir.is_dir():
            continue

        env_config = subdir / "env_config.yaml"
        if not env_config.exists():
            no_config += 1
            continue

        try:
            if migrate_env_config_file(env_config, dry_run=args.dry_run):
                migrated += 1
                action = "Would migrate" if args.dry_run else "Migrated"
                print(f"[OK] {action}: {subdir.name}")
            else:
                skipped += 1
        except Exception as e:
            print(f"[FAIL] {subdir.name}: {e}")

    print("")
    print("==========================================")
    suffix = " (dry run)" if args.dry_run else ""
    print(f"Done{suffix}: {migrated} migrated, {skipped} already up-to-date, {no_config} no env_config.yaml")
    print("==========================================")


if __name__ == "__main__":
    main()