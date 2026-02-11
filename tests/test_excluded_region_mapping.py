"""
Test script for analyzing map_excluded_regions() fallback behavior.

This script tests how often each fallback case in map_excluded_regions() is triggered
for different numbers of selected regions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from src.data.preprocessor import RawDataLoader, DataSelector, DataProcessor


def analyze_map_excluded_regions_cases(
    selected_region_ids: list,
    order_region_ids: pd.Series,
    warehouse_to_region_df: pd.DataFrame
) -> dict:
    """
    Analyzes which case each excluded region falls into.
    
    Returns:
        dict with keys:
            - 'no_pairs': Count of excluded regions with no warehouse-region pairs
            - 'no_matching_warehouses': Count of excluded regions with pairs but no matching warehouses
            - 'cost_based_mapping': Count of excluded regions mapped using cost-based method
            - 'total_excluded': Total number of excluded regions
            - 'case_breakdown': Dict mapping excluded region ID to case name
    """
    stats = {
        'no_pairs': 0,
        'no_matching_warehouses': 0,
        'cost_based_mapping': 0,
        'total_excluded': 0,
        'case_breakdown': {}
    }
    
    # Convert to string and set for comparison
    selected_region_ids_str = [str(rid) for rid in selected_region_ids]
    selected_region_ids_set = set(str(rid) for rid in selected_region_ids)
    
    # Find excluded regions
    excluded_mask = ~order_region_ids.astype(str).isin(selected_region_ids_set)
    excluded_regions = order_region_ids[excluded_mask].unique()
    stats['total_excluded'] = len(excluded_regions)
    
    # For each excluded region, determine which case it falls into
    for excluded_region in excluded_regions:
        excluded_region_str = str(excluded_region)
        
        # Get all warehouse-region pairs involving this excluded region
        excluded_pairs = warehouse_to_region_df[
            warehouse_to_region_df['destinationregionid'].astype(str) == excluded_region_str
        ]
        
        # Case 1: No warehouse-region pairs found
        if len(excluded_pairs) == 0:
            stats['no_pairs'] += 1
            stats['case_breakdown'][excluded_region_str] = 'no_pairs'
        
        # Case 2 & 3: Warehouse-region pairs found
        else:
            warehouse_ids = excluded_pairs['sourcenodeid'].unique()
            included_pairs = warehouse_to_region_df[
                (warehouse_to_region_df['destinationregionid'].astype(str).isin(selected_region_ids_str)) &
                (warehouse_to_region_df['sourcenodeid'].isin(warehouse_ids))
            ]
            
            # Case 2: No matching warehouses
            if len(included_pairs) == 0:
                stats['no_matching_warehouses'] += 1
                stats['case_breakdown'][excluded_region_str] = 'no_matching_warehouses'
            
            # Case 3: Matching warehouses found, use cost-based mapping
            else:
                stats['cost_based_mapping'] += 1
                stats['case_breakdown'][excluded_region_str] = 'cost_based_mapping'
    
    return stats


def test_map_excluded_regions_fallbacks():
    """
    Test how often each fallback case is triggered for different n_regions values.
    """
    print("\n" + "=" * 70)
    print("MAP_EXCLUDED_REGIONS FALLBACK ANALYSIS")
    print("=" * 70)
    
    # Load raw data
    raw_data_path = Path("data_files/raw")
    loader = RawDataLoader(raw_data_path)
    loader.load_all()
    loader.validate_relationships()
    
    print(f"\n[INFO] Loaded data:")
    print(f"  Total regions: {len(loader.regions_df)}")
    print(f"  Total orders: {len(loader.orders_df)}")
    print(f"  Unique regions in orders: {loader.orders_df['regionid'].nunique()}")
    
    # Get available region IDs
    available_region_ids = loader.regions_df['regionid'].unique().tolist()
    total_regions = len(available_region_ids)
    
    # Test different n_regions values
    # Test with small, medium, and large selections relative to total regions
    n_regions_values = [
        3, 5, 10, 15, 20, 30, 40, 50, total_regions  # Include all regions as edge case
    ]
    # Filter to only values <= total_regions
    n_regions_values = [n for n in n_regions_values if n <= total_regions]
    
    # Use fixed seed for reproducibility
    seed = 42
    
    results = []
    
    for n_regions in n_regions_values:
        print(f"\n{'=' * 70}")
        print(f"Testing with n_regions = {n_regions} (out of {total_regions} total)")
        print(f"{'=' * 70}")
        
        # Select regions
        selector = DataSelector(
            n_skus=100,  # Not used for this test
            n_warehouses=10,  # Not used for this test
            n_regions=n_regions,
            selection_seed=seed
        )
        selector.select_regions(available_region_ids)
        
        # Create a minimal DataProcessor to access the mapping logic
        # We only need selected_region_ids and warehouse_to_region_df
        processor = DataProcessor(
            selected_sku_ids=[],  # Not used
            selected_warehouse_ids=[],  # Not used
            selected_region_ids=selector.selected_region_ids,
            warehouse_to_region_df=loader.warehouse_to_region_df,
            orders_df=loader.orders_df,
            order_sku_demand_df=loader.order_sku_demand_df,
            skus_df=loader.skus_df,
            regions_df=loader.regions_df,
        )
        
        # Get order region IDs
        order_region_ids = loader.orders_df['regionid']
        
        # Analyze cases
        stats = analyze_map_excluded_regions_cases(
            selector.selected_region_ids,
            order_region_ids,
            loader.warehouse_to_region_df
        )
        
        # Store results
        results.append({
            'n_regions': n_regions,
            'total_regions': total_regions,
            'selected_regions': len(selector.selected_region_ids),
            'excluded_regions': stats['total_excluded'],
            'no_pairs': stats['no_pairs'],
            'no_matching_warehouses': stats['no_matching_warehouses'],
            'cost_based_mapping': stats['cost_based_mapping'],
        })
        
        # Print statistics
        print(f"\n[RESULTS]")
        print(f"  Selected regions: {len(selector.selected_region_ids)}")
        print(f"  Excluded regions: {stats['total_excluded']}")
        
        if stats['total_excluded'] > 0:
            print(f"\n  Fallback case distribution:")
            print(f"    Case 1 (no warehouse-region pairs): {stats['no_pairs']} ({100*stats['no_pairs']/stats['total_excluded']:.1f}%)")
            print(f"    Case 2 (no matching warehouses): {stats['no_matching_warehouses']} ({100*stats['no_matching_warehouses']/stats['total_excluded']:.1f}%)")
            print(f"    Case 3 (cost-based mapping): {stats['cost_based_mapping']} ({100*stats['cost_based_mapping']/stats['total_excluded']:.1f}%)")
            
            # Show some examples
            print(f"\n  Example excluded regions by case:")
            case_examples = defaultdict(list)
            for region_id, case in stats['case_breakdown'].items():
                case_examples[case].append(region_id)
            
            for case_name, examples in case_examples.items():
                print(f"    {case_name}: {examples[:5]}")  # Show first 5 examples
        else:
            print(f"  No excluded regions (all regions selected)")
    
    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"{'n_regions':<12} {'excluded':<12} {'no_pairs':<12} {'no_match':<12} {'cost_based':<12}")
    print(f"{'-' * 70}")
    
    for r in results:
        print(f"{r['n_regions']:<12} {r['excluded_regions']:<12} {r['no_pairs']:<12} "
              f"{r['no_matching_warehouses']:<12} {r['cost_based_mapping']:<12}")
    
    # Print percentage summary
    print(f"\n{'=' * 70}")
    print("PERCENTAGE BREAKDOWN (of excluded regions)")
    print(f"{'=' * 70}")
    print(f"{'n_regions':<12} {'excluded':<12} {'no_pairs %':<12} {'no_match %':<12} {'cost_based %':<12}")
    print(f"{'-' * 70}")
    
    for r in results:
        if r['excluded_regions'] > 0:
            no_pairs_pct = 100 * r['no_pairs'] / r['excluded_regions']
            no_match_pct = 100 * r['no_matching_warehouses'] / r['excluded_regions']
            cost_based_pct = 100 * r['cost_based_mapping'] / r['excluded_regions']
            print(f"{r['n_regions']:<12} {r['excluded_regions']:<12} {no_pairs_pct:<12.1f} "
                  f"{no_match_pct:<12.1f} {cost_based_pct:<12.1f}")
        else:
            print(f"{r['n_regions']:<12} {r['excluded_regions']:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    print(f"\n{'=' * 70}")
    print("TEST COMPLETED")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    test_map_excluded_regions_fallbacks()