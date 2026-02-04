"""
Data preprocessing module for real-world data integration.

This module provides classes for loading raw data files, selecting subsets,
processing orders, and extracting shipment costs for use in the environment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.random import SeedSequence

if TYPE_CHECKING:
    from src.config.schema import DataSplitConfig
    from src.environment.context import ShipmentCosts


@dataclass
class PreprocessedData:
    """
    Implements a container for preprocessed demand data with optional train/validation split.

    Attributes:
        demand_data (pd.DataFrame): Preprocessed demand data for training (or full data if no split).
        val_demand_data (Optional[pd.DataFrame]): Preprocessed demand data for validation. None if no validation data is available.
    """

    # Demand data
    demand_data: pd.DataFrame  
    val_demand_data: Optional[pd.DataFrame] = None 


class RawDataLoader:
    """
    Implements a raw data loader that loads and validates raw data files from the data_files/raw directory.
    """
    
    def __init__(self, raw_data_path: str):
        """
        Initializes the raw data loader.
        
        Args:
            raw_data_path (str): Path to the directory containing raw CSV files.
        """

        # Store raw data path
        self.raw_data_path = Path(raw_data_path)
        
        # DataFrames to be populated by load_all()
        self.warehouses_df: Optional[pd.DataFrame] = None
        self.regions_df: Optional[pd.DataFrame] = None
        self.warehouse_to_region_df: Optional[pd.DataFrame] = None
        self.suppliers_df: Optional[pd.DataFrame] = None
        self.supplier_to_warehouse_df: Optional[pd.DataFrame] = None
        self.skus_df: Optional[pd.DataFrame] = None
        self.skus_per_supplier_df: Optional[pd.DataFrame] = None
        self.orders_df: Optional[pd.DataFrame] = None
        self.order_sku_demand_df: Optional[pd.DataFrame] = None
    
    def load_all(self):
        """
        Loads all raw CSV files from the raw data directory.
        """

        # Load all 9 files and store them as DataFrames
        self.warehouses_df = pd.read_csv(self.raw_data_path / "01_warehouses.csv")
        self.regions_df = pd.read_csv(self.raw_data_path / "02_regions.csv")
        self.warehouse_to_region_df = pd.read_csv(self.raw_data_path / "03_warehouse_to_region.csv")
        self.suppliers_df = pd.read_csv(self.raw_data_path / "04_suppliers.csv")
        self.supplier_to_warehouse_df = pd.read_csv(self.raw_data_path / "05_supplier_to_warehouse.csv")
        self.skus_df = pd.read_csv(self.raw_data_path / "06_skus.csv")
        self.skus_per_supplier_df = pd.read_csv(self.raw_data_path / "07_skus_per_supplier.csv")
        self.orders_df = pd.read_csv(self.raw_data_path / "08_orders.csv")
        self.order_sku_demand_df = pd.read_csv(self.raw_data_path / "09_order_sku_demand.csv")
    
    def validate_relationships(self):
        """
        Validates referential integrity of the loaded data by checking that:  
            - Orders reference valid regions (via country codes)
            - Order-SKU demands reference valid orders and SKUs
            - Warehouse-region relationships reference valid warehouses and regions
        """

        # Validate orders have regionid column and reference valid regions
        if self.orders_df is not None and self.regions_df is not None:
            if 'regionid' not in self.orders_df.columns:
                raise ValueError(
                    "Orders DataFrame must have 'regionid' column. "
                    "Run scripts/preprocess_orders_add_regionid.py first."
                )
            valid_region_ids = set(self.regions_df['regionid'].unique())
            order_region_ids = set(self.orders_df['regionid'].dropna().unique())
            invalid_regions = order_region_ids - valid_region_ids
            if invalid_regions:
                raise ValueError(f"Orders reference invalid region IDs: {len(invalid_regions)} regions")
        
        # Validate order-SKU demands reference valid orders
        if self.order_sku_demand_df is not None and self.orders_df is not None:
            valid_order_ids = set(self.orders_df['salesorderid'].unique())
            demand_order_ids = set(self.order_sku_demand_df['salesorderid'].unique())
            invalid_orders = demand_order_ids - valid_order_ids
            if invalid_orders:
                raise ValueError(f"Order-SKU demands reference invalid orders: {len(invalid_orders)} orders")
        
        # Validate order-SKU demands reference valid SKUs
        if self.order_sku_demand_df is not None and self.skus_df is not None:
            valid_sku_ids = set(self.skus_df['itemid'].unique())
            demand_sku_ids = set(self.order_sku_demand_df['itemid'].unique())
            invalid_skus = demand_sku_ids - valid_sku_ids
            if invalid_skus:
                raise ValueError(f"Order-SKU demands reference invalid SKUs: {len(invalid_skus)} SKUs")


class DataSelector:
    """
    Selects subsets of SKUs, warehouses, and regions based on configuration by uniformly selecting random IDs.  
    """
    
    def __init__(self, n_skus: int, n_warehouses: int, n_regions: int, selection_seed: Optional[int] = None):
        """
        Initializes the data selector.
        
        Args:
            n_skus (int): Target number of SKUs to select.
            n_warehouses (int): Target number of warehouses to select.
            n_regions (int): Target number of regions to select.
            selection_seed (Optional[int]): Random seed for reproducible data selection. Defaults to None.
        """

        # Store target number of SKUs, warehouses, and regions
        self.n_skus = n_skus
        self.n_warehouses = n_warehouses
        self.n_regions = n_regions
        
        # Selected IDs (to be populated by selection methods)
        self.selected_sku_ids: Optional[List[str]] = None
        self.selected_warehouse_ids: Optional[List[str]] = None
        self.selected_region_ids: Optional[List[str]] = None
        self.selected_supplier_ids: Optional[List[str]] = None
        
        # Initialize RNG
        self._rng = np.random.default_rng(selection_seed)
    
    def select_skus(self, available_sku_ids: List[str]):
        """
        Selects n_skus SKUs uniformly at random from the available SKU IDs.
        
        Args:
            available_sku_ids (List[str]): List of available SKU IDs to select from.
        """

        # Check if there are enough available SKUs
        if len(available_sku_ids) < self.n_skus:
            raise ValueError(
                f"Cannot select {self.n_skus} SKUs from {len(available_sku_ids)} available SKUs"
            )
        
        # Select n_skus SKUs uniformly at random
        self.selected_sku_ids = self._rng.choice(
            available_sku_ids, size=self.n_skus, replace=False
        ).tolist()
    
    def select_warehouses(self, available_warehouse_ids: List[str]):
        """
        Selects n_warehouses warehouses uniformly at random from the available warehouse IDs.
        
        Args:
            available_warehouse_ids (List[str]): List of available warehouse IDs to select from.
        """

        # Check if there are enough available warehouses
        if len(available_warehouse_ids) < self.n_warehouses:
            raise ValueError(
                f"Cannot select {self.n_warehouses} warehouses from {len(available_warehouse_ids)} available warehouses"
            )
        
        # Select n_warehouses warehouses uniformly at random
        self.selected_warehouse_ids = self._rng.choice(
            available_warehouse_ids, size=self.n_warehouses, replace=False
        ).tolist()
    
    def select_regions(self, available_region_ids: List[str]):
        """
        Selects n_regions regions uniformly at random from the available region IDs.
        
        Args:
            available_region_ids (List[str]): List of available region IDs to select from.
        """

        # Check if there are enough available regions
        if len(available_region_ids) < self.n_regions:
            raise ValueError(
                f"Cannot select {self.n_regions} regions from {len(available_region_ids)} available regions"
            )
        
        # Select n_regions regions uniformly at random
        self.selected_region_ids = self._rng.choice(
            available_region_ids, size=self.n_regions, replace=False
        ).tolist()
    
    def select_suppliers(
        self,
        selected_sku_ids: List[str],
        skus_per_supplier_df: pd.DataFrame,
    ):
        """
        Selects the first supplier for each selected SKU (first supplier that appears in skus_per_supplier_df).
        
        Args:
            selected_sku_ids: List of selected SKU IDs.
            skus_per_supplier_df: DataFrame with columns: itemid, supplierid
        """

        # Initialize mapping from SKU ID to first supplier ID
        sku_to_supplier = {}

        # Iterate over skus_per_supplier_df to build mapping from SKU ID to first supplier ID
        for _, row in skus_per_supplier_df.iterrows():
            # Get SKU ID and supplier ID
            sku_id = str(row['itemid'])
            supplier_id = str(row['supplierid'])

            # Store the first supplier ID we encounter for each SKU
            if sku_id not in sku_to_supplier:
                sku_to_supplier[sku_id] = supplier_id
        
        # Initialize set of selected supplier IDs
        selected_supplier_ids = []
        
        # Iterate over selected SKU IDs to select the corresponding supplier
        for sku_id in selected_sku_ids:
            supplier_id = sku_to_supplier.get(str(sku_id))
            if supplier_id is None:
                raise ValueError(f"No suppliers found for SKU {sku_id}")
            selected_supplier_ids.append(supplier_id)
        self.selected_supplier_ids = selected_supplier_ids
    

class DataSplitter:
    """
    Splits demand data into training and validation subsets.
    Supports ratio-based and explicit timestep-based splitting.
    """
    
    @staticmethod
    def split_by_ratio(
        data: pd.DataFrame, 
        train_ratio: float, 
        seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data by timestep ratio (e.g., 80% train, 20% val).
        Ensures no overlap between train and validation timesteps.
        
        Args:
            data (pd.DataFrame): DataFrame with 'timestep' column
            train_ratio (float): Ratio of data for training (0.0-1.0)
            seed (Optional[int]): Optional seed for reproducibility (not used for ratio split, but kept for API consistency)
        
        Returns:
            train_data (pd.DataFrame): Training subset DataFrame
            val_data (pd.DataFrame): Validation subset DataFrame
        """
        
        # Get unique timesteps and sort
        unique_timesteps = sorted(data['timestep'].unique())
        if len(unique_timesteps) == 0:
            raise ValueError("Data contains no timesteps")
        
        # Calculate split point
        split_idx = int(len(unique_timesteps) * train_ratio)
        
        # Ensure at least one timestep in each split
        if split_idx == 0:
            raise ValueError(f"train_ratio ({train_ratio}) results in 0 training timesteps")
        if split_idx >= len(unique_timesteps):
            raise ValueError(f"train_ratio ({train_ratio}) results in 0 validation timesteps")

        # Split timesteps into train and validation timesteps
        train_timesteps = unique_timesteps[:split_idx]
        val_timesteps = unique_timesteps[split_idx:]
        
        # Split data into train and validation data
        train_data = data[data['timestep'].isin(train_timesteps)].copy()
        val_data = data[data['timestep'].isin(val_timesteps)].copy()
        
        return train_data, val_data
    
    @staticmethod
    def split_by_timesteps(
        data: pd.DataFrame,
        train_timesteps: List[int],
        val_timesteps: List[int]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data by explicit timestep lists.
        Validates no overlap between lists.
        
        Args:
            data (pd.DataFrame): DataFrame with 'timestep' column
            train_timesteps (List[int]): List of timesteps for training
            val_timesteps (List[int]): List of timesteps for validation
        
        Returns:
            train_data (pd.DataFrame): Training subset DataFrame
            val_data (pd.DataFrame): Validation subset DataFrame
        """
        
        # Convert timestep lists to sets for faster lookup
        train_set = set(train_timesteps)
        val_set = set(val_timesteps)

        # Check if train and validation timesteps exist in data
        available_timesteps = set(data['timestep'].unique())
        train_missing = train_set - available_timesteps
        val_missing = val_set - available_timesteps
        if train_missing:
            raise ValueError(
                f"train_timesteps contains timesteps not in data: {sorted(train_missing)}"
            )
        if val_missing:
            raise ValueError(
                f"val_timesteps contains timesteps not in data: {sorted(val_missing)}"
            )
        
        # Split data into train and validation data
        train_data = data[data['timestep'].isin(train_timesteps)].copy()
        val_data = data[data['timestep'].isin(val_timesteps)].copy()
        
        return train_data, val_data


class DataProcessor:
    """
    Processes and filters data based on selected subsets by mapping excluded regions to included regions, 
    filtering to selected SKUs and regions, mapping original IDs to 0-indexed IDs, and aggregating by timestep.
    """
    
    def __init__(
        self,
        selected_sku_ids: List[str],
        selected_warehouse_ids: List[str],
        selected_region_ids: List[str],
        selected_supplier_ids: List[str],
        warehouse_to_region_df: pd.DataFrame,
        supplier_to_warehouse_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        order_sku_demand_df: pd.DataFrame,
        skus_df: pd.DataFrame,
        regions_df: pd.DataFrame,
    ):
        """
        Initializes the data processor.
        
        Args:
            selected_sku_ids (List[str]): Selected SKU IDs.
            selected_warehouse_ids (List[str]): Selected warehouse IDs.
            selected_region_ids (List[str]): Selected region IDs.
            selected_supplier_ids (List[str]): Selected supplier IDs.
            warehouse_to_region_df (pd.DataFrame): Warehouse-to-region relationships with costs.
            supplier_to_warehouse_df (pd.DataFrame): Supplier-to-warehouse relationships with costs.
            orders_df (pd.DataFrame): Orders DataFrame (must have 'regionid' column).
            order_sku_demand_df (pd.DataFrame): Order-SKU demand DataFrame.
            skus_df (pd.DataFrame): SKUs DataFrame (must have 'sku_index' column).
            regions_df (pd.DataFrame): Regions DataFrame (must have 'region_index' column).
        """

        # Store general parameters
        self.selected_sku_ids = selected_sku_ids
        self.selected_warehouse_ids = selected_warehouse_ids
        self.selected_region_ids = selected_region_ids
        self.selected_supplier_ids = selected_supplier_ids
        self.warehouse_to_region_df = warehouse_to_region_df
        self.supplier_to_warehouse_df = supplier_to_warehouse_df
        self.orders_df = orders_df
        self.order_sku_demand_df = order_sku_demand_df
        self.skus_df = skus_df
        self.regions_df = regions_df
    
    def map_excluded_regions(self, order_region_ids: pd.Series) -> pd.Series:
        """
        Reassigns orders from regions not in `selected_region_ids` to the most suitable
        included region based on shared warehouse connections and minimum average fixed cost.
        If no suitable mapping exists, a fallback included region is used.

        Args:
            order_region_ids (pd.Series): Region IDs from orders (may include excluded regions).
            
        Returns:
            pd.Series: Mapped region IDs (all in selected_region_ids).
        """

        # Copy order region IDs
        mapped_regions = order_region_ids.copy()
        
        # Convert to string and set for comparison
        selected_region_ids_str = [str(rid) for rid in self.selected_region_ids]
        selected_region_ids_set = set(str(rid) for rid in self.selected_region_ids)
        
        # Find excluded regions
        excluded_mask = ~order_region_ids.astype(str).isin(selected_region_ids_set)
        excluded_regions = order_region_ids[excluded_mask].unique()
        
        # For each excluded region, find most similar included region
        for excluded_region in excluded_regions:
            # Convert to string for comparison
            excluded_region_str = str(excluded_region)
            
            # Get all warehouse-region pairs involving this excluded region
            excluded_pairs = self.warehouse_to_region_df[
                self.warehouse_to_region_df['destinationregionid'].astype(str) == excluded_region_str
            ]
            
            # No warehouse-region pairs found, use first included region as fallback
            if len(excluded_pairs) == 0:
                nearest_region = self.selected_region_ids[0]

            # If warehouse-region pairs found, find included regions that have relationships with same warehouses
            else:
                warehouse_ids = excluded_pairs['sourcenodeid'].unique()
                included_pairs = self.warehouse_to_region_df[
                    (self.warehouse_to_region_df['destinationregionid'].astype(str).isin(selected_region_ids_str)) &
                    (self.warehouse_to_region_df['sourcenodeid'].isin(warehouse_ids))
                ]

                # No matching warehouses, use first included region
                if len(included_pairs) == 0:
                    nearest_region = self.selected_region_ids[0]

                # If matching warehouses found, find region with minimum average distance/cost
                else:
                    region_costs = included_pairs.groupby('destinationregionid')['fixed_costs'].mean()
                    nearest_region_str = str(region_costs.idxmin())
                    nearest_region = next((rid for rid in self.selected_region_ids if str(rid) == nearest_region_str), self.selected_region_ids[0])
            
            # Map all orders from the current excluded region to the nearest included region
            mapped_regions[order_region_ids.astype(str) == excluded_region_str] = nearest_region
        
        return mapped_regions
    
    def get_updated_shipment_costs(self) -> 'ShipmentCosts':
        """
        Extracts both outbound and inbound shipment costs (fixed and variable) for all pairs in the selected sets.
        If a pair does not exist in the data, it uses the average cost as fallback.
        
        Returns:
            shipment_costs (ShipmentCosts): ShipmentCosts dataclass containing:
                - outbound_fixed_per_order: Shape (n_warehouses, n_regions)
                - outbound_variable_per_weight: Shape (n_warehouses, n_regions)
                - inbound_fixed_per_order: Shape (n_suppliers, n_warehouses)
                - inbound_variable_per_weight: Shape (n_suppliers, n_warehouses)
        """

        # Import ShipmentCosts dataclass
        from src.environment.context import ShipmentCosts

        # Get number of warehouses, regions, and suppliers
        n_warehouses = len(self.selected_warehouse_ids)
        n_regions = len(self.selected_region_ids)
        n_suppliers = len(self.selected_supplier_ids)
        
        # Initialize outbound cost matrices (warehouse -> region)
        outbound_fixed = np.zeros((n_warehouses, n_regions), dtype=float)
        outbound_variable = np.zeros((n_warehouses, n_regions), dtype=float)
        
        # Extract outbound costs for each warehouse-region pair
        for wh_idx, warehouse_id in enumerate(self.selected_warehouse_ids):
            # Convert warehouse ID to string for comparison
            warehouse_id_str = str(warehouse_id)

            for reg_idx, region_id in enumerate(self.selected_region_ids):
                # Convert region ID to string for comparison
                region_id_str = str(region_id)
                
                # Get data for this warehouse-region pair
                pair_data = self.warehouse_to_region_df[
                    (self.warehouse_to_region_df['sourcenodeid'].astype(str) == warehouse_id_str) &
                    (self.warehouse_to_region_df['destinationregionid'].astype(str) == region_id_str)
                ]
                
                # If pair exists, use costs from the data
                if len(pair_data) > 0:
                    outbound_fixed[wh_idx, reg_idx] = pair_data['fixed_costs'].iloc[0]
                    outbound_variable[wh_idx, reg_idx] = pair_data['variable_costs_per_weight'].iloc[0]
                
                # If pair doesn't exist, use fallback method
                else:
                    # Get all costs for this warehouse
                    warehouse_costs = self.warehouse_to_region_df[
                        self.warehouse_to_region_df['sourcenodeid'].astype(str) == warehouse_id_str
                    ]
                    
                    # If no costs for this warehouse exist, use high default for fixed, zero for variable
                    if len(warehouse_costs) == 0:
                        outbound_fixed[wh_idx, reg_idx] = 10000.0
                        outbound_variable[wh_idx, reg_idx] = 0.0

                    # If other costs for this warehouse exist, use mean cost
                    else:
                        outbound_fixed[wh_idx, reg_idx] = warehouse_costs['fixed_costs'].mean()
                        outbound_variable[wh_idx, reg_idx] = warehouse_costs['variable_costs_per_weight'].mean()
        
        # Initialize inbound cost matrices (supplier -> warehouse)
        inbound_fixed = np.zeros((n_suppliers, n_warehouses), dtype=float)
        inbound_variable = np.zeros((n_suppliers, n_warehouses), dtype=float)

        # Extract inbound costs for each supplier-warehouse pair
        for supp_idx, supplier_id in enumerate(self.selected_supplier_ids):
            # Convert supplier ID to string for comparison
            supplier_id_str = str(supplier_id)
            
            for wh_idx, warehouse_id in enumerate(self.selected_warehouse_ids):
                # Convert warehouse ID to string for comparison
                warehouse_id_str = str(warehouse_id)
                
                # Get data for this supplier-warehouse pair
                pair_data = self.supplier_to_warehouse_df[
                    (self.supplier_to_warehouse_df['sourcesupplierid'].astype(str) == supplier_id_str) &
                    (self.supplier_to_warehouse_df['destinationnodeid'].astype(str) == warehouse_id_str)
                ]
                
                # If pair exists, use costs from the data
                if len(pair_data) > 0:
                    inbound_fixed[supp_idx, wh_idx] = pair_data['fixed_costs'].iloc[0]
                    inbound_variable[supp_idx, wh_idx] = pair_data['variable_costs_per_weight'].iloc[0]
                
                # If pair doesn't exist, use fallback method
                else:
                    # Get all costs for this supplier
                    supplier_costs = self.supplier_to_warehouse_df[
                        self.supplier_to_warehouse_df['sourcesupplierid'].astype(str) == supplier_id_str
                    ]
                    
                    # If no costs for this supplier exist, use high default for fixed, zero for variable
                    if len(supplier_costs) == 0:
                        inbound_fixed[supp_idx, wh_idx] = 10000.0
                        inbound_variable[supp_idx, wh_idx] = 0.0
                    
                    # If other costs for this supplier exist, use mean cost
                    else:
                        inbound_fixed[supp_idx, wh_idx] = supplier_costs['fixed_costs'].mean()
                        inbound_variable[supp_idx, wh_idx] = supplier_costs['variable_costs_per_weight'].mean()
        
        return ShipmentCosts(
            outbound_fixed=outbound_fixed,
            outbound_variable=outbound_variable,
            inbound_fixed=inbound_fixed,
            inbound_variable=inbound_variable
        )
    
    def get_distances(self) -> np.ndarray:
        """
        Extracts the distances (in km) for all (warehouse, region) pairs in the selected sets.
        If a pair does not exist in the data, it uses the average distance for the warehouse as fallback.
        
        Returns:
            distance_matrix (np.ndarray): Distance matrix. Shape: (n_warehouses, n_regions).
        """
        
        # Initialize distance matrix
        n_warehouses = len(self.selected_warehouse_ids)
        n_regions = len(self.selected_region_ids)
        distance_matrix = np.zeros((n_warehouses, n_regions), dtype=float)
        
        # Extract distances for each warehouse-region pair
        for wh_idx, warehouse_id in enumerate(self.selected_warehouse_ids):
            # Convert warehouse ID to string for comparison
            warehouse_id_str = str(warehouse_id)

            for reg_idx, region_id in enumerate(self.selected_region_ids):
                # Convert region ID to string for comparison
                region_id_str = str(region_id)
                
                # Get data for this warehouse-region pair
                pair_data = self.warehouse_to_region_df[
                    (self.warehouse_to_region_df['sourcenodeid'].astype(str) == warehouse_id_str) &
                    (self.warehouse_to_region_df['destinationregionid'].astype(str) == region_id_str)
                ]
                
                # If pair exists, use distance from the data
                if len(pair_data) > 0:
                    distance_matrix[wh_idx, reg_idx] = pair_data['distance_km'].iloc[0]
                
                # If pair doesn't exist, use fallback method
                else:
                    # Get all distances for this warehouse
                    warehouse_distances = self.warehouse_to_region_df[
                        self.warehouse_to_region_df['sourcenodeid'].astype(str) == warehouse_id_str
                    ]
                    
                    # If no distances for this warehouse exist, use high default
                    if len(warehouse_distances) == 0:
                        distance_matrix[wh_idx, reg_idx] = 10000.0

                    # If distances for this warehouse exist, use mean distance
                    else:
                        mean_distance = warehouse_distances['distance_km'].mean()
                        distance_matrix[wh_idx, reg_idx] = mean_distance
        
        return distance_matrix
    
    def get_sku_weights(self) -> np.ndarray:
        """
        Extracts SKU weights for selected SKUs in the same order as selected_sku_ids.
        
        Returns:
            sku_weights (np.ndarray): SKU weights array. Shape: (n_skus,).
        """
        
        # Check if 'weight' column exists in skus_df
        if 'weight' not in self.skus_df.columns:
            raise ValueError("SKUs DataFrame must have 'weight' column")
        
        # Create mapping from SKU ID to weight
        sku_id_to_weight = dict(zip(self.skus_df['itemid'], self.skus_df['weight']))
        
        # Extract weights for selected SKUs in order
        sku_weights = np.array([
            sku_id_to_weight[sku_id] 
            for sku_id in self.selected_sku_ids
        ], dtype=float)
        
        return sku_weights
    
    def create_processed_demand_data(self) -> pd.DataFrame:
        """
        Creates processed demand DataFrame by performing the following steps:
            1. Merging dataframes of orders with order-SKU demand
            2. Mapping excluded regions to included regions based on minimum average fixed cost
            3. Filtering only selected SKUs and regions
            4. Mapping global string IDs to local selection indices for correct indexing later.
            5. Aggregating by timestep.
        
        Returns:
            processed_demand_data (pd.DataFrame): Processed demand data with columns:
                timestep, region_id, order_id, sku_id, quantity
        """
        #print(f"[DEBUG] Total number of orders before merge: {len(self.order_sku_demand_df)}")
        # 1. Merge orders with order-SKU demand
        merged = self.order_sku_demand_df.merge(
            self.orders_df,
            on='salesorderid',
            how='inner'
        )
        #print(f"[DEBUG] Total number of orders after merge: {len(merged)}")
        #print(f"DEBUG: Unique day_ids before SKU filtering: {merged['day_id'].nunique()}")

        # 2. Map excluded regions to included regions
        merged['regionid'] = self.map_excluded_regions(merged['regionid'])

        #print(f"DEBUG: Orders after SKU filtering: {len(merged)}")
        #print(f"DEBUG: Unique day_ids after SKU filtering: {merged['day_id'].nunique()}")  
        
        # 3. Filter only selected SKUs
        merged = merged[merged['itemid'].isin(self.selected_sku_ids)]
        #print(f"DEBUG: Orders after selecting SKUs: {len(merged)}")
        #print(f"DEBUG: Unique day_ids after selecting SKUs: {merged['day_id'].nunique()}")
        
        # 4. Build mappings from global string IDs (e.g. SKU_12345, REG_EU) to global CSV indices
        sku_global_id_to_index = dict(zip(self.skus_df['itemid'], self.skus_df['sku_index']))
        region_global_id_to_index = dict(zip(self.regions_df['regionid'], self.regions_df['region_index']))
        
        # 5. Build mappings from global CSV indices to local selection indices
        sku_global_to_selection_index = {
            sku_global_id_to_index[sku_id]: idx 
            for idx, sku_id in enumerate(self.selected_sku_ids)
        }
        region_global_to_selection_index = {
            region_global_id_to_index[region_id]: idx 
            for idx, region_id in enumerate(self.selected_region_ids)
        }
        
        # 6. Convert global string IDs to global CSV indices to local selection indices for correct indexing later.
        merged['sku_csv_index'] = merged['itemid'].map(sku_global_id_to_index)
        merged['region_csv_index'] = merged['regionid'].map(region_global_id_to_index)
        merged['sku_selection_index'] = merged['sku_csv_index'].map(sku_global_to_selection_index)
        merged['region_selection_index'] = merged['region_csv_index'].map(region_global_to_selection_index)
        merged = merged.drop(columns=['sku_csv_index', 'region_csv_index'])
        
        # 7. Create final DataFrame with required columns
        processed_demand_data = pd.DataFrame({
            'timestep': merged['day_id'].astype(int),
            'region_id': merged['region_selection_index'].astype(int),
            'order_id': merged['salesorderid'],
            'sku_id': merged['sku_selection_index'].astype(int),
            'quantity': merged['quantity'].astype(float)
        })
        
        #print(f"DEBUG: Final processed data timesteps: {processed_demand_data['timestep'].nunique()}")
        #print(f"DEBUG: Final unique timesteps: {sorted(processed_demand_data['timestep'].unique())}")

        # 8. Sort by timestep for efficient querying
        processed_demand_data = processed_demand_data.sort_values(['timestep', 'region_id', 'order_id', 'sku_id']).reset_index(drop=True)
        
        return processed_demand_data
    

class DataPreprocessor:
    """
    Implements a data preprocessor that orchestrates the loading of raw data, selecting subsets, and processing data 
    to create preprocessed data ready for environment use.
    """
    
    def __init__(self, raw_data_path: str, n_skus: int, n_warehouses: int, n_regions: int):
        """
        Initializes the data preprocessor.
        
        Args:
            raw_data_path (str): Path to the directory containing the raw CSV files.
            n_skus (int): Number of SKUs to select.
            n_warehouses (int): Number of warehouses to select.
            n_regions (int): Number of regions to select.
        """

        # Store general parameters
        self.raw_data_path = raw_data_path
        self.n_skus = n_skus
        self.n_warehouses = n_warehouses
        self.n_regions = n_regions
    
    def preprocess(
        self, 
        data_split: Optional['DataSplitConfig'] = None,
        seed: Optional[int] = None
    ) -> Tuple[PreprocessedData, 'ShipmentCosts', np.ndarray, np.ndarray]:
        """
        Implements the raw data preprocessing pipeline by loading raw data, selecting subsets of SKUs, 
        warehouses, and regions according to the configuration, and processing the data to create 
        preprocessed data ready for environment use. If a seed is provided, it is used to ensure reproducible 
        subset selection. If data_split is provided, splits the data into train and validation sets.
        
        Args:
            data_split (Optional[DataSplitConfig]): Optional data split configuration. If provided, splits
                the processed data into train and validation sets. Defaults to None.
            seed (Optional[int]): Random seed for reproducibile data selection. Defaults to None.

        
        Returns:
            preprocessed_data (PreprocessedData): PreprocessedData instance containing demand data.
                If data_split is None: preprocessed_data.demand_data contains all data and
                preprocessed_data.val_demand_data is None.
                If data_split is provided: preprocessed_data.demand_data contains training data,
                preprocessed_data.val_demand_data contains validation data.
            shipment_costs (ShipmentCosts): ShipmentCosts dataclass containing outbound and inbound shipment costs.
            sku_weights (np.ndarray): SKU unit weights. Shape: (n_skus,).
            distances (np.ndarray): Distance matrix. Shape: (n_warehouses, n_regions).
        """

        # 1. Load and validate raw data
        loader = RawDataLoader(self.raw_data_path)
        loader.load_all()
        loader.validate_relationships()

        # 2. Get available IDs of SKUs, warehouses, and regions
        available_sku_ids = loader.order_sku_demand_df['itemid'].unique().tolist()
        available_warehouse_ids = loader.warehouses_df['nodeid'].unique().tolist()
        available_region_ids = loader.regions_df['regionid'].unique().tolist()

        # 3. Select subsets of SKUs, warehouses, regions, and suppliers      
        selector = DataSelector(self.n_skus, self.n_warehouses, self.n_regions, seed)
        selector.select_skus(available_sku_ids)
        selector.select_warehouses(available_warehouse_ids)
        selector.select_regions(available_region_ids)
        selector.select_suppliers(selector.selected_sku_ids, loader.skus_per_supplier_df)

        #print(f"[DEBUG] Selected SKUs: {selector.selected_sku_ids}")
        #print(f"[DEBUG] Selected warehouses: {selector.selected_warehouse_ids}")
        #print(f"[DEBUG] Selected regions: {selector.selected_region_ids}")
        #print(f"[DEBUG] Selected suppliers: {selector.selected_supplier_ids}")

        # 4. Process data
        processor = DataProcessor(
            selected_sku_ids=selector.selected_sku_ids,
            selected_warehouse_ids=selector.selected_warehouse_ids,
            selected_region_ids=selector.selected_region_ids,
            selected_supplier_ids=selector.selected_supplier_ids,
            warehouse_to_region_df=loader.warehouse_to_region_df,
            supplier_to_warehouse_df=loader.supplier_to_warehouse_df,
            orders_df=loader.orders_df,
            order_sku_demand_df=loader.order_sku_demand_df,
            skus_df=loader.skus_df,
            regions_df=loader.regions_df,
        )
        processed_data = processor.create_processed_demand_data()
        shipment_costs = processor.get_updated_shipment_costs()
        sku_weights = processor.get_sku_weights()
        distances = processor.get_distances()

        # 5. Split data if data_split config is provided
        if data_split is not None:
            if data_split.type == "ratio":
                train_data, val_data = DataSplitter.split_by_ratio(
                    processed_data, 
                    train_ratio=data_split.train_ratio,
                    seed=seed
                )
            elif data_split.type == "explicit":
                train_data, val_data = DataSplitter.split_by_timesteps(
                    processed_data,
                    train_timesteps=data_split.train_timesteps,
                    val_timesteps=data_split.val_timesteps
                )
            else:
                raise ValueError(f"Unknown data_split type: {data_split.type}")
            
            preprocessed_data = PreprocessedData(
                demand_data=train_data,
                val_demand_data=val_data
            )
        else:
            preprocessed_data = PreprocessedData(
                demand_data=processed_data,
                val_demand_data=None
            )

        return preprocessed_data, shipment_costs, sku_weights, distances

