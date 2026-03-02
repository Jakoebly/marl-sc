#!/usr/bin/env python3
"""
Standalone test script for DataGenerator.

Run from project root:
    python scripts/test_data_generator.py

Adjust RAW_DATA_PATH and MODELS_PATH below if your paths differ.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# ---------------------------------------------------------------------------
# Paths — adjust these if your setup differs (e.g. match config_files/environments/base_env.yaml)
# ---------------------------------------------------------------------------
RAW_DATA_PATH = project_root / "data_files" / "raw"
MODELS_PATH = project_root / "src" / "data" / "notebooks" / "outputs" / "generated_models"


def main():
    from src.data.data_generator import DataGenerator
    from src.utils.seed_manager import SeedManager, EXPERIMENT_SEEDS

    print("=" * 60)
    print("DataGenerator Test Script")
    print("=" * 60)
    print(f"Raw data path: {RAW_DATA_PATH}")
    print(f"Models path:   {MODELS_PATH}")
    print()

    # Check paths exist
    if not RAW_DATA_PATH.exists():
        print(f"[WARN] Raw data path does not exist: {RAW_DATA_PATH}")
        print("       Some tests may fail.")
    if not MODELS_PATH.exists():
        print(f"[WARN] Models path does not exist: {MODELS_PATH}")
        print("       DataGenerator will fail.")
        print()
        return

    n_warehouses = 3
    n_skus = 5
    n_regions = 4

    # -----------------------------------------------------------------------
    # 1. Run with SeedManager (seeded)
    # -----------------------------------------------------------------------
    print("-" * 60)
    print("1. With SeedManager (seed=42)")
    print("-" * 60)

    seed_manager = SeedManager(root_seed=42, seed_registry=EXPERIMENT_SEEDS)
    gen = DataGenerator(
        raw_data_path=str(RAW_DATA_PATH),
        models_path=str(MODELS_PATH),
        seed_manager=seed_manager,
    )

    try:
        data = gen.generate(n_warehouses, n_skus, n_regions)
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return

    print("Keys:", list(data.keys()))
    print()
    print("penalty_cost:", data["penalty_cost"])
    print("sku_weights:", data["sku_weights"])
    print("distances shape:", len(data["distances"]), "x", len(data["distances"][0]) if data["distances"] else 0)
    print("outbound_fixed shape:", len(data["outbound_fixed"]), "x", len(data["outbound_fixed"][0]) if data["outbound_fixed"] else 0)
    print("outbound_variable shape:", len(data["outbound_variable"]), "x", len(data["outbound_variable"][0]) if data["outbound_variable"] else 0)
    print("inbound_fixed shape:", len(data["inbound_fixed"]), "x", len(data["inbound_fixed"][0]) if data["inbound_fixed"] else 0)
    print("inbound_variable shape:", len(data["inbound_variable"]), "x", len(data["inbound_variable"][0]) if data["inbound_variable"] else 0)
    print()

    # -----------------------------------------------------------------------
    # 2. Reproducibility: same seed produces same output
    # -----------------------------------------------------------------------
    print("-" * 60)
    print("2. Reproducibility: same seed produces same output")
    print("-" * 60)

    seed_manager_2 = SeedManager(root_seed=42, seed_registry=EXPERIMENT_SEEDS)
    gen_2 = DataGenerator(
        raw_data_path=str(RAW_DATA_PATH),
        models_path=str(MODELS_PATH),
        seed_manager=seed_manager_2,
    )
    data_2 = gen_2.generate(n_warehouses, n_skus, n_regions)

    all_match = True
    for key in data:
        if data[key] != data_2[key]:
            print(f"[FAIL] {key} differs between runs")
            all_match = False
    if all_match:
        print("[OK] All outputs identical for seed=42")
    print()

    # -----------------------------------------------------------------------
    # 3. Different seed produces different output
    # -----------------------------------------------------------------------
    print("-" * 60)
    print("3. Different seed produces different output")
    print("-" * 60)

    seed_manager_3 = SeedManager(root_seed=99, seed_registry=EXPERIMENT_SEEDS)
    gen_3 = DataGenerator(
        raw_data_path=str(RAW_DATA_PATH),
        models_path=str(MODELS_PATH),
        seed_manager=seed_manager_3,
    )
    data_3 = gen_3.generate(n_warehouses, n_skus, n_regions)

    any_diff = False
    for key in data:
        if data[key] != data_3[key]:
            any_diff = True
            break
    if any_diff:
        print("[OK] Output differs for seed=99")
    else:
        print("[WARN] Output identical for seed=99 (unlikely but possible)")
    print()

    # -----------------------------------------------------------------------
    # 4. Without SeedManager (unseeded)
    # -----------------------------------------------------------------------
    print("-" * 60)
    print("4. Without SeedManager (unseeded)")
    print("-" * 60)

    gen_unseeded = DataGenerator(
        raw_data_path=str(RAW_DATA_PATH),
        models_path=str(MODELS_PATH),
        seed_manager=None,
    )
    data_unseeded = gen_unseeded.generate(n_warehouses, n_skus, n_regions)
    print("Keys:", list(data_unseeded.keys()))
    print("penalty_cost:", data_unseeded["penalty_cost"])
    print("sku_weights (first 3):", data_unseeded["sku_weights"][:3])
    print("[OK] Unseeded run completed")
    print()

    # -----------------------------------------------------------------------
    # 5. Shape sanity checks
    # -----------------------------------------------------------------------
    print("-" * 60)
    print("5. Shape sanity checks")
    print("-" * 60)

    expected_distances = (n_warehouses, n_regions)
    expected_outbound = (n_warehouses, n_regions)
    expected_inbound = (n_warehouses, n_skus)

    dist_shape = (len(data["distances"]), len(data["distances"][0]) if data["distances"] else 0)
    out_fix_shape = (len(data["outbound_fixed"]), len(data["outbound_fixed"][0]) if data["outbound_fixed"] else 0)
    in_fix_shape = (len(data["inbound_fixed"]), len(data["inbound_fixed"][0]) if data["inbound_fixed"] else 0)

    checks = [
        ("distances", dist_shape, expected_distances),
        ("outbound_fixed", out_fix_shape, expected_outbound),
        ("inbound_fixed", in_fix_shape, expected_inbound),
    ]
    for name, got, expected in checks:
        if got == expected:
            print(f"[OK] {name}: {got}")
        else:
            print(f"[FAIL] {name}: got {got}, expected {expected}")

    print()
    print("penalty_cost len:", len(data["penalty_cost"]), f"(expected {n_skus})")
    print("sku_weights len:", len(data["sku_weights"]), f"(expected {n_skus})")
    print()
    print("=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
