import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.seed_manager import SeedManager

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

# Constants
EARTH_RADIUS_KM = 6371.0
PENALTY_MIN, PENALTY_MAX = 8.6, 15.2

# =============================================================================
# Penalty cost helpers (from legacy cost_generator.py)
# =============================================================================

def _smooth_factor(i: int, j: int, amp: float = 0.06) -> float:
    """Small bounded deterministic variation in [1-amp, 1+amp]."""
    return 1.0 + amp * math.sin(0.9 * i + 1.7 * j + 0.3)

def _bounded_increasing(
    lo: float, hi: float, n: int, amp: float = 0.02, seed: int = 0,
) -> np.ndarray:
    """Generate *n* strictly increasing values in [lo, hi] with log-spacing."""
    if n == 1:
        return np.array([math.sqrt(lo * hi)])
    vals = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    for k in range(1, n - 1):
        vals[k] *= _smooth_factor(k, seed, amp=amp)
    vals = np.clip(vals, lo, hi)
    for k in range(1, n):
        if vals[k] <= vals[k - 1]:
            vals[k] = vals[k - 1] + (hi - lo) * 1e-4
    return vals

# =============================================================================
# Weight Generator
# =============================================================================

class WeightGenerator:
    """
    Implements a class for generating synthetic SKU weights from a pre-fitted 
    log-normal mixture model exported from weight_generator.ipynb as ``weight_gmm.pkl``.

    Attributes:
        model_path (Path): Path to the pre-fitted log-normal mixture model.
        rng (np.random.Generator): Random number generator.
    """

    def __init__(self, model_path: Path, rng: np.random.Generator):
        """
        Initializes the weight generator.
        
        Args:
            model_path (Path): Path to the pre-fitted log-normal mixture model.
            rng (np.random.Generator): Random number generator.
        """

        # Store random number generator and model parameters
        self._rng = rng
        self._params = self._load_model(model_path)["full_mix_params"]

    def generate(self, n_skus: int) -> np.ndarray:
        """
        Generates *n_skus* synthetic weights sorted ascending (lightest first).
        
        Args:
            n_skus (int): Number of synthetic weights to generate.
            
        Returns:
            weights (np.ndarray): Synthetic weights sorted ascending (lightest first).
        """
        
        # Sample from the truncated log-normal mixture	
        raw = self._sample_truncated_lognormal_mixture(n_skus)

        # Sort the weights ascending
        weights = np.sort(raw)

        return weights
    
    def _sample_truncated_lognormal_mixture(self, size: int) -> np.ndarray:
        """
        Samples from the truncated log-normal mixture using hierarchical inverse-CDF sampling.
        
        Args:
            size (int): Number of samples to generate.
            
        Returns:
            samples (np.ndarray): Samples from the truncated log-normal mixture.
        """

        # Get model parameters
        p = self._params

        # Sample components from the mixture
        weights = p["mix_weights"]
        components = self._rng.choice(len(weights), size=size, p=weights)

        # Sample from the mixture using the sampled components
        samples = np.empty(size)
        for k in range(len(weights)):
            mask = components == k
            n_k = mask.sum()
            if n_k > 0:
                samples[mask] = self._sample_truncated_lognormal(
                    p["means"][k], p["stds"][k], p["w_min"], p["w_max"], n_k,
                )
        return samples

    def _sample_truncated_lognormal(
        self, mu: float, sigma: float, lo: float, hi: float, size: int,
    ) -> np.ndarray:
        """
        Samples from LogNormal(mu, sigma) truncated to [lo, hi] via inverse CDF.
        
        Args:
            mu (float): Mean of the log-normal distribution.
            sigma (float): Standard deviation of the log-normal distribution.
            lo (float): Lower bound of the truncated distribution.
            hi (float): Upper bound of the truncated distribution.
            size (int): Number of samples to generate.
        """
        # Compute the inverse CDF of the log-normal distribution
        Phi = stats.norm.cdf
        Phi_inv = stats.norm.ppf

        # Convert truncation bounds
        log_lo, log_hi = np.log(lo), np.log(hi)

        # Compute the CDF values at the standardized truncation bounds
        a = Phi((log_lo - mu) / sigma)
        b = Phi((log_hi - mu) / sigma)

        # Sample uniformly in probability space between a and b
        u = self._rng.uniform(a, b, size=size)

        # Map uniform samples back through the inverse normal CDF:
        z = Phi_inv(u)

        # Map back to the original scale
        samples = np.exp(mu + sigma * z)

        return samples

    @staticmethod
    def _load_model(path: Path):
        """
        Loads a pre-fitted model from the given path.
        
        Args:
            path (Path): Path to the pre-fitted model.
            
        Returns:
            model (dict): Pre-fitted model.
        """

        # Load the model from the given path
        with open(path, "rb") as f:
            model =pickle.load(f)

        return model

# =============================================================================
# Distance Generator
# =============================================================================

class LocationDataGenerator :
    """
    Implements a class for generating synthetic location data (distances and max_wh_capacity )
    from real-world geographic data. Uses a pre-fitted geographic GMM (``distance_gmm_geo.pkl``) 
    for region density scoring and raw CSV data for selecting regions, warehouses, 
    and suppliers.

    Attributes:
        model_path (Path): Path to the pre-fitted geographic GMM.
        raw_data_path (Path): Path to the raw geographic data.
        rng (np.random.Generator): Random number generator.
    """
    
    def __init__(self, model_path: Path, raw_data_path: Path, rng: np.random.Generator):
        """
        Initializes the distance generator.
        
        Args:
            model_path (Path): Path to the pre-fitted geographic GMM.
            raw_data_path (Path): Path to the raw geographic data.
        """

        # Store random number generator and model parameters
        self._rng = rng
        self._raw_data_path = Path(raw_data_path)
        self._gmm_geo = self._load_model(model_path)

    def generate(self, n_regions: int, n_warehouses: int, n_skus: int) -> dict:
        """
        Generates outbound ``(n_warehouses, n_regions)`` and inbound ``(n_warehouses, n_skus)`` distances.
        
        Args:
            n_regions (int): Number of regions to generate distances for.
            n_warehouses (int): Number of warehouses to generate distances for.
            n_skus (int): Number of SKUs to generate distances for.
            
        Returns:
            distances (dict): Dictionary containing outbound and inbound distances.
        """

        # Load raw data
        raw = self._load_raw_data()

        # Sample SKU indices
        sku_ids = self._sample_skus(raw, n_skus)

        # Enrich regions with demand intensity and geographic density
        self._enrich_regions(raw, sku_ids)

        # Compute sampling probabilities for regions
        probs = self._compute_sampling_probabilities(raw["regions_df"])

        # Sample regions and warehouses
        selected_regions = self._sample_regions(raw, probs, n_regions)
        selected_warehouses = self._select_warehouses(raw, selected_regions, n_warehouses)

        # Compute outbound distances
        outbound = self._compute_outbound_distances(selected_warehouses, selected_regions)

        # Find closest suppliers and compute inbound distances (including the respective supplier IDs)
        inbound, closest_supplier_ids = self._find_closest_suppliers(raw, selected_warehouses, sku_ids)

        # Get lead times for the (warehouse, closest-supplier) pairs
        lead_times = self._get_lead_times(raw, selected_warehouses, sku_ids, closest_supplier_ids)

        # Compute warehouse capacities
        max_wh_capacities = self._compute_warehouse_capacities(selected_warehouses)

        # Store the location data
        location_data = {
            "outbound_distances": outbound,
            "inbound_distances": inbound,
            "lead_times": lead_times,
            "max_wh_capacities": max_wh_capacities,
        }

        return location_data

    def _load_raw_data(self) -> dict:
        """
        Loads raw data from the given path.
        
        Args:
            path (Path): Path to the raw data.
            
        Returns:
            raw (dict): Raw data.
        """

        # Get the raw data path
        p = self._raw_data_path

        # Load the raw data from the raw data path
        raw_data = {
            "regions_df": pd.read_csv(p / "02_regions.csv"),
            "warehouses_df": pd.read_csv(p / "01_warehouses.csv"),
            "suppliers_df": pd.read_csv(p / "04_suppliers.csv"),
            "skus_df": pd.read_csv(p / "06_skus.csv"),
            "skus_per_supplier_df": pd.read_csv(p / "07_skus_per_supplier.csv"),
            "supplier_to_warehouse_df": pd.read_csv(p / "05_supplier_to_warehouse.csv"),
            "orders_df": pd.read_csv(p / "08_orders.csv"),
            "order_sku_demand_df": pd.read_csv(p / "09_order_sku_demand.csv"),
        }

        return raw_data

    def _sample_skus(self, raw: dict, n_skus: int) -> np.ndarray:
        """
        Samples n_skus distinct SKU indices without replacement.
        """

        # Get all SKU IDs
        all_sku_ids = raw["skus_df"]["itemid"].values

        # Check if n_skus exceeds available SKUs
        if n_skus > len(all_sku_ids):
            raise ValueError(f"n_skus={n_skus} exceeds available SKUs ({len(all_sku_ids)})")
        
        # Sample n_skus distinct SKU indices without replacement
        indices = self._rng.choice(len(all_sku_ids), size=n_skus, replace=False)

        # Store the sampled SKU indices
        sampled_sku_ids = all_sku_ids[indices]

        return sampled_sku_ids

    def _enrich_regions(self, raw: dict, sku_ids: np.ndarray) -> None:
        """
        Attaches demand intensity and geographic density columns to regions_df.
        
        Args:
            raw (dict): Raw data.
            sku_ids (np.ndarray): Sampled SKU indices.
        """

        # Get the regions DataFrame
        regions_df = raw["regions_df"]

        # Compute demand intensities and attach them to the regions DataFrame
        intensities = self._compute_demand_intensities(raw, sku_ids)
        regions_df["intensity"] = intensities

        # Normalize the demand intensities and attach them to the regions DataFrame
        total = intensities.sum()
        regions_df["intensity_normalized"] = intensities / total if total > 0 else 1.0 / len(regions_df)

        # Compute geographic scores and attach them to the regions DataFrame
        geo_scores = self._compute_geographic_scores(regions_df)
        regions_df["p_geo"] = geo_scores

    def _compute_demand_intensities(self, raw: dict, sku_ids: np.ndarray) -> np.ndarray:
        """
        Computes weight-based demand intensity I_r per region (unnormalized).
        
        Args:
            raw (dict): Raw data.
            sku_ids (np.ndarray): Sampled SKU indices.
        """

        # Merge the order-SKU demand DataFrame with the orders DataFrame 
        merged = raw["order_sku_demand_df"].merge(
            raw["orders_df"][["salesorderid", "day_id", "regionid"]],
            on="salesorderid", how="inner",
        )

        # Filter for the sampled SKU indices and create a lookup map for the SKU weights
        merged = merged[merged["itemid"].isin(sku_ids)]
        sku_weight_map = raw["skus_df"].set_index("itemid")["weight"].to_dict()

        # Compute the total weight of the demand for each region
        merged["total_weight"] = merged["quantity"] * merged["itemid"].map(sku_weight_map).fillna(1.0)

        # Compute the demand intensity for each region
        n_periods = max(merged["day_id"].nunique(), 1)
        region_intensity = merged.groupby("regionid")["total_weight"].sum() / n_periods

        # Get all region IDs
        all_regions = raw["regions_df"]["regionid"].values

        # Store demand intensities in an array
        intensities = np.array([region_intensity.get(r, 0.0) for r in all_regions])

        return intensities

    def _compute_geographic_scores(self, regions_df: pd.DataFrame) -> np.ndarray:
        """
        Scores each region via the fitted geographic GMM log-likelihood.
        
        Args:
            regions_df (pd.DataFrame): Regions DataFrame.
            
        Returns:
            scores (np.ndarray): Geographic scores.
        """

        # Convert latitude and longitude to 3D unit vectors
        coords_3d = self._latlon_to_unit_vector(
            regions_df["latitude"].values, regions_df["longitude"].values,
        )

        # Compute the log-likelihood of the geographic GMM
        log_density = self._gmm_geo.score_samples(coords_3d)

        # Normalize the log-likelihood
        scores = np.exp(log_density - log_density.max())

        return scores

    def _compute_sampling_probabilities(
        self, regions_df: pd.DataFrame, alpha: float = 0.15,
    ) -> np.ndarray:
        """
        Nudges uniform distribution towards intensity-weighted distribution for region sampling.
        
        Args:
            regions_df (pd.DataFrame): Regions DataFrame.
            alpha (float): Nudging factor.

        Returns:
            probs (np.ndarray): Sampling probabilities.
        """

        # Compute the raw weights
        raw_weights = regions_df["p_geo"].values * regions_df["intensity_normalized"].values

        # Compute the intensity-weighted weights
        total = raw_weights.sum()
        p_intensity = raw_weights / total if total > 0 else np.ones(len(regions_df)) / len(regions_df)

        # Compute the uniform weights
        n = len(regions_df)
        p_uniform = np.ones(n) / n

        # Nudge the uniform weights towards the intensity-weighted weights using the nudging factor alpha
        probs = (1 - alpha) * p_uniform + alpha * p_intensity

        # Normalize the probabilities
        probs /= probs.sum()

        return probs

    def _sample_regions(
        self, raw: dict, probs: np.ndarray, n_regions: int,
    ) -> pd.DataFrame:
        """
        Samples n_regions distinct regions without replacement.
        
        Args:
            raw (dict): Raw data.
            probs (np.ndarray): Sampling probabilities.
            n_regions (int): Number of regions to sample.
        """

        # Get the regions DataFrame
        regions_df = raw["regions_df"]

        # Check if n_regions exceeds available regions
        if n_regions > len(regions_df):
            raise ValueError(f"n_regions={n_regions} exceeds available regions ({len(regions_df)})")
        
        # Sample n_regions distinct regions without replacement using given probabilities
        indices = self._rng.choice(len(regions_df), size=n_regions, replace=False, p=probs)

        # Store the sampled regions
        sampled_regions = regions_df.iloc[indices].copy().reset_index(drop=True)

        return sampled_regions

    def _select_warehouses(
        self, raw: dict, selected_regions: pd.DataFrame, n_warehouses: int,
    ) -> pd.DataFrame:
        """
        Selects n_warehouses warehouses via demand-weighted greedy k-median.
        
        Args:
            raw (dict): Raw data.
            selected_regions (pd.DataFrame): Selected regions.
            n_warehouses (int): Number of warehouses to select.
        """

        # Get the warehouses DataFrame
        warehouses_df = raw["warehouses_df"]

        # Check if n_warehouses exceeds available warehouses
        if n_warehouses > len(warehouses_df):
            raise ValueError(
                f"n_warehouses={n_warehouses} exceeds available warehouses ({len(warehouses_df)})"
            )

        # Compute the haversine distance matrix between the warehouses and the selected region coordinates
        wh_coords = warehouses_df[["latitude", "longitude"]].values
        reg_coords = selected_regions[["latitude", "longitude"]].values
        dist_matrix = self._haversine_distance_matrix(wh_coords, reg_coords)

        # Compute weights for the selected regions
        intensities = selected_regions.get("intensity")
        if intensities is None or intensities.sum() <= 0:
            # If no intensity data, use uniform weights
            omega = np.ones(len(selected_regions)) / len(selected_regions)
        else:
            # If intensity data, use intensity-based weights
            omega = intensities.values / intensities.values.sum()

        # Select the warehouses via intensity-weighted greedy k-median
        selected_indices = self._greedy_k_median(dist_matrix, omega, n_warehouses)

        # Store the selected warehouses
        selected_warehouses = warehouses_df.iloc[selected_indices].copy().reset_index(drop=True)

        return selected_warehouses

    def _compute_outbound_distances(
        self, warehouses: pd.DataFrame, regions: pd.DataFrame,
    ) -> np.ndarray:
        """
        Computes the outbound distance matrix between selected warehouses and regions.
        
        Args:
            warehouses (pd.DataFrame): Warehouses DataFrame.
            regions (pd.DataFrame): Regions DataFrame.
        """

        # Compute the haversine distance matrix between the warehouses and the regions
        dist_matrix = self._haversine_distance_matrix(
            warehouses[["latitude", "longitude"]].values,
            regions[["latitude", "longitude"]].values,
        )

        return dist_matrix

    def _find_closest_suppliers(
        self, raw: dict, warehouses: pd.DataFrame, sku_ids: np.ndarray,
    ) -> tuple:
        """
        Finds the geographically closest feasible supplier for each (warehouse, SKU) pair.
        Returns both the haversine distance to that supplier and its ID so that
        downstream consumers can reuse the same selection without recomputing it.

        Args:
            raw (dict): Raw data.
            warehouses (pd.DataFrame): Warehouses DataFrame.
            sku_ids (np.ndarray): Sampled SKU indices.

        Returns:
            inbound_distances (np.ndarray): Distance (km) to the closest supplier. Shape (n_warehouses, n_skus)
            closest_supplier_ids (np.ndarray): Object array of supplier IDs. Shape (n_warehouses, n_skus)
        """

        # Create a lookup map for the SKU-supplier links
        sku_supplier_map = (
            raw["skus_per_supplier_df"]
            .groupby("itemid")["supplierid"]
            .apply(list)
            .to_dict()
        )

        # Create a lookup map for the supplier coordinates
        supplier_coord_map = (
            raw["suppliers_df"]
            .set_index("supplierid")[["latitude", "longitude"]]
            .to_dict(orient="index")
        )

        # Get the number of warehouses and SKUs
        n_warehouses = len(warehouses)
        n_skus = len(sku_ids)

        # Get the warehouse coordinates
        wh_coords = warehouses[["latitude", "longitude"]].values

        # Initialize the inbound distances matrix
        inbound_distances = np.full((n_warehouses, n_skus), np.inf)
        closest_supplier_ids = np.full((n_warehouses, n_skus), None, dtype=object)

        # Loop over SKUs to compute the inbound distances for each (warehouse, SKU)
        for sku_idx, sku_id in enumerate(sku_ids):
            # Get the feasible suppliers for the current SKU
            feasible = sku_supplier_map.get(sku_id, [])

            # If no feasible suppliers, skip
            if not feasible:
                continue

            # If feasible but no coordinates, skip
            feasible_with_coords = [
                sid for sid in feasible if sid in supplier_coord_map
            ]
            if not feasible_with_coords:
                continue

            # Get the supplier coordinates for the feasible suppliers
            sup_coords = np.array([
                [supplier_coord_map[sid]["latitude"],
                 supplier_coord_map[sid]["longitude"]]
                for sid in feasible_with_coords
            ])

            # Compute the haversine distance matrix between the warehouse and the supplier coordinates
            dists = self._haversine_distance_matrix(wh_coords, sup_coords)
            
            # Find the closest supplier for each warehouse
            closest_idx = dists.argmin(axis=1)
            inbound_distances[:, sku_idx] = dists[np.arange(n_warehouses), closest_idx]

            for wh_idx in range(n_warehouses):
                closest_supplier_ids[wh_idx, sku_idx] = feasible_with_coords[closest_idx[wh_idx]]

        return inbound_distances, closest_supplier_ids

    def _get_lead_times(
        self,
        raw: dict,
        warehouses: pd.DataFrame,
        sku_ids: np.ndarray,
        closest_supplier_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Looks up the lead time for each (warehouse, SKU) pair using the closest 
        supplier determined by _find_closest_suppliers.

        Args:
            raw (dict): Raw data.
            warehouses (pd.DataFrame): Selected warehouses.
            sku_ids (np.ndarray): Sampled SKU IDs.
            closest_supplier_ids (np.ndarray): Object array of closest supplier IDs. Shape (n_warehouses, n_skus)

        Returns:
            lead_times (np.ndarray): Lead time in days. Shape (n_warehouses, n_skus)
        """

        # Get the supplier to warehouse DataFrame
        stw_df = raw["supplier_to_warehouse_df"]

        # Create a lookup map for the lead time for each (supplier, warehouse) pair
        lane_lt_map: dict = {}
        for _, row in stw_df.iterrows():
            key = (row["sourcesupplierid"], row["destinationnodeid"])
            lane_lt_map[key] = row["shippingtime_days"]

        # Create a lookup map for the median lead time for each supplier (used as fallback)
        supplier_median_lt: dict = (
            stw_df.groupby("sourcesupplierid")["shippingtime_days"]
            .median()
            .to_dict()
        )

        # Get the number of warehouses and SKUs
        n_warehouses = len(warehouses)
        n_skus = len(sku_ids)
        warehouse_ids = warehouses["nodeid"].values

        # Initialize the lead times matrix
        lead_times = np.full((n_warehouses, n_skus), np.nan)

        # Loop over warehouses and SKUs to compute the lead times
        for wh_idx in range(n_warehouses):
            # Get the warehouse ID
            wh_id = warehouse_ids[wh_idx]

            # Loop over SKUs to compute the lead time for each (warehouse, SKU) pair
            for sku_idx in range(n_skus):
                # Get the closest supplier ID and skip if no supplier exists
                sup_id = closest_supplier_ids[wh_idx, sku_idx]
                if sup_id is None:
                    continue

                # Get the lead time for the (supplier, warehouse) pair
                lt = lane_lt_map.get((sup_id, wh_id))

                # If the lead time exists, store it
                if lt is not None:
                    lead_times[wh_idx, sku_idx] = lt
                # If the lead time does not exist, use the median lead time for the supplier as fallback
                else:
                    fallback = supplier_median_lt.get(sup_id)
                    if fallback is not None:
                        lead_times[wh_idx, sku_idx] = fallback

        return lead_times

    def _compute_warehouse_capacities(self, warehouses: pd.DataFrame) -> np.ndarray:
        """
        Computes the maximum capacities of the warehouses.
        
        Args:
            warehouses (pd.DataFrame): Warehouses DataFrame.

        Returns:
            capacities (np.ndarray): Maximum capacities of the warehouses.
        """
        
        # Get the warehouse capacities
        max_wh_capacities = warehouses["weight"].values

        return max_wh_capacities

    @staticmethod
    def _load_model(path: Path):
        """
        Loads a pre-fitted model from the given path.
        
        Args:
            path (Path): Path to the pre-fitted model.
            
        Returns:
            model (dict): Pre-fitted model.
        """

        # Load the model from the given path
        with open(path, "rb") as f:
            model =pickle.load(f)

        return model

    @staticmethod
    def _haversine_distance_matrix(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
        """
        Computes pairwise haversine distances (km) between two (lat, lon) arrays.
        
        Args:
            coords_a (np.ndarray): Coordinates of the first set of points.
            coords_b (np.ndarray): Coordinates of the second set of points.

        Returns:
            dist_matrix (np.ndarray): Haversine distance matrix.
        """
        
        # Convert the coordinates to radians
        lat_a = np.radians(coords_a[:, 0])[:, None]
        lon_a = np.radians(coords_a[:, 1])[:, None]
        lat_b = np.radians(coords_b[:, 0])[None, :]
        lon_b = np.radians(coords_b[:, 1])[None, :]
        
        # Compute the differences in latitude and longitude
        dlat = lat_b - lat_a
        dlon = lon_b - lon_a

        # Compute the haversine distances
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        dist_matrix = EARTH_RADIUS_KM * c

        return dist_matrix

    @staticmethod
    def _latlon_to_unit_vector(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
        """
        Converts lat/lon (degrees) to 3-D unit vectors on the sphere.
        
        Args:
            lat_deg (np.ndarray): Latitude in degrees.
            lon_deg (np.ndarray): Longitude in degrees.

        Returns:
            unit_vector (np.ndarray): 3-D unit vector.
        """
        
        # Clip the latitude and longitude to the valid range
        lat = np.clip(np.asarray(lat_deg, dtype=float), -90.0, 90.0)
        lon = ((np.asarray(lon_deg, dtype=float) + 180.0) % 360.0) - 180.0

        # Convert the latitude and longitude to radians
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        # Compute the x, y, and z components of the unit vector
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        # Stack the x, y, and z components into a 3-D unit vector
        unit_vector = np.column_stack([x, y, z])

        return unit_vector

    @staticmethod
    def _greedy_k_median(
        dist_matrix: np.ndarray, weights: np.ndarray, k: int,
    ) -> np.ndarray:
        """
        Performs greedy forward-selection for the demand-weighted k-median problem.
        
        Args:
            dist_matrix (np.ndarray): Distance matrix.
            weights (np.ndarray): Weights.
            k (int): Number of warehouses to select.

        Returns:
            selected (np.ndarray): Selected warehouses.
        """

        # Initialize the selected and remaining warehouse indices
        selected: list[int] = []
        remaining = set(range(dist_matrix.shape[0]))
        
        # Loop over the number of warehouses
        for _ in range(k):
            # Track the best objective value and candidate warehouse index 
            best_cost = np.inf
            best_idx = -1
            
            # Loop over the remaining warehouses to find the best addition
            for idx in remaining:
                # Build the trial set by adding the current candidate
                trial = selected + [idx]

                # Compute the minimum distance from each region to the selected warehouses in the trial set
                min_dists = dist_matrix[trial, :].min(axis=0)

                # Compute the objective value of the trial set
                cost = np.dot(weights, min_dists)

                # Update the best cost and index if the trial set improves the objective value
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
                
            # Add the best candidate to the selected set and remove it from the remaining set
            selected.append(best_idx)
            remaining.discard(best_idx)

        return np.array(selected)


# =============================================================================
# Cost Generator
# =============================================================================

class CostGenerator:
    """Generates inbound/outbound costs conditioned on distances via pre-fitted 3-D GMMs.
    Uses ``costs_gmm_outbound.pkl`` and ``costs_gmm_inbound.pkl`` exported from
    cost_generator.ipynb.  Clipping bounds are computed from the raw lane-cost
    data at the percentiles given by *clip_percentile*.

    Attributes:
        outbound_model_path (Path): Path to the outbound GMM model.
        inbound_model_path (Path): Path to the inbound GMM model.
        raw_data_path (Path): Path to the raw data.
        rng (np.random.Generator): Random number generator.
        eps (float): Epsilon for numerical stability.
        clip_percentile (tuple): Percentiles for clipping bounds.
    """

    def __init__(
        self,
        outbound_model_path: Path,
        inbound_model_path: Path,
        raw_data_path: Path,
        rng: np.random.Generator,
        eps: float = 1e-6,
        clip_percentile: tuple = (0, 100),
    ):
        """
        Initializes the cost generator.
        
        Args:
            outbound_model_path (Path): Path to the outbound GMM model.
            inbound_model_path (Path): Path to the inbound GMM model.
            raw_data_path (Path): Path to the raw data.
            rng (np.random.Generator): Random number generator.
            eps (float): Epsilon for numerical stability.
            clip_percentile (tuple): Percentiles for clipping bounds.
        """

        # Store random number generator and model parameters
        self._rng = rng
        self._eps = eps
        self._raw_data_path = Path(raw_data_path)
        self._gmm_outbound = self._load_model(outbound_model_path)
        self._gmm_inbound = self._load_model(inbound_model_path)
        self._clip_bounds = self._compute_clipping_bounds(clip_percentile)

    def generate(
        self, outbound_distances: np.ndarray, inbound_distances: np.ndarray,
    ) -> dict:
        """
        Generates outbound and inbound fixed/variable cost matrices matching the distance shapes.
        
        Args:
            outbound_distances (np.ndarray): Outbound distance matrix.
            inbound_distances (np.ndarray): Inbound distance matrix.
        """
        
        # Get the shapes of the distance matrices
        shape_out = outbound_distances.shape
        shape_in = inbound_distances.shape

        # Sample costs for outbound distances
        cf_out, cv_out = self._sample_conditional_costs(
            outbound_distances.ravel(), self._gmm_outbound, self._clip_bounds["outbound"],
        )

        # Sample costs for inbound distances
        cf_in, cv_in = self._sample_conditional_costs(
            inbound_distances.ravel(), self._gmm_inbound, self._clip_bounds["inbound"],
        )

        # Store the costs in a dictionary
        costs = {
            "outbound_fixed": cf_out.reshape(shape_out),
            "outbound_variable": cv_out.reshape(shape_out),
            "inbound_fixed": cf_in.reshape(shape_in),
            "inbound_variable": cv_in.reshape(shape_in),
        }

        return costs

    def _compute_clipping_bounds(self, clip_percentile: tuple) -> dict:
        """
        Computes cost clipping bounds from the real lane-cost data.
        
        Args:
            clip_percentile (tuple): Percentiles for clipping bounds.

        Returns:
            clipping_bounds (dict): Clipping bounds.
        """

        # Get the high percentile
        _, hi_pct = clip_percentile

        # Load the outbound and inbound data
        outbound_df = pd.read_csv(self._raw_data_path / "03_warehouse_to_region.csv")
        inbound_df = pd.read_csv(self._raw_data_path / "05_supplier_to_warehouse.csv")

        # Compute the clipping bounds for the outbound and inbound data
        clipping_bounds ={
            "outbound": {
                "clip_lo": 0.0,
                "clip_hi_fix": float(np.percentile(outbound_df["fixed_costs"].values, hi_pct)),
                "clip_hi_var": float(np.percentile(outbound_df["variable_costs_per_weight"].values, hi_pct)),
            },
            "inbound": {
                "clip_lo": 0.0,
                "clip_hi_fix": float(np.percentile(inbound_df["fixed_costs"].values, hi_pct)),
                "clip_hi_var": float(np.percentile(inbound_df["variable_costs_per_weight"].values, hi_pct)),
            },
        }

        return clipping_bounds

    def _sample_conditional_costs(
        self, distances: np.ndarray, gmm, clip: dict,
    ) -> tuple:
        """
        Generates correlated (c_fix, c_var) for each distance via conditional GMM sampling.
        
        Args:
            distances (np.ndarray): Distances.
            gmm (dict): GMM model.
            clip (dict): Clipping bounds.

        Returns:
            c_fix (np.ndarray): Fixed costs.
        """

        # Convert the distances to a 1D array
        distances = np.asarray(distances, dtype=float).ravel()

        # Get parameters
        n = len(distances)
        K = gmm.n_components
        eps = self._eps

        # Initialize the fixed and variable costs
        c_fix = np.empty(n)
        c_var = np.empty(n)

        # Loop over the given distances to sample the costs
        for i in range(n):
            # Transform distance to log space
            x = np.log1p(distances[i])

            # Compute the unnormalized posterior probabilities
            unnorm = np.array([
                gmm.weights_[k] * norm.pdf(
                    x, loc=gmm.means_[k, 0],
                    scale=np.sqrt(gmm.covariances_[k, 0, 0] + eps),
                )
                for k in range(K)
            ])

            # Normalize the posterior probabilities
            total = unnorm.sum()
            posteriors = np.ones(K) / K if total < 1e-300 else unnorm / total

            # Sample the component index
            k = self._rng.choice(K, p=posteriors)

            # Get the mean and covariance of the selected component
            mu = gmm.means_[k]
            S = gmm.covariances_[k]

            # Compute the conditional mean and covariance
            S_xx_inv = 1.0 / (S[0, 0] + eps)
            diff = x - mu[0]
            mu_cond = mu[1:] + S[1:, 0] * S_xx_inv * diff
            cov_cond = S[1:, 1:] - np.outer(S[1:, 0], S[0, 1:]) * S_xx_inv
            cov_cond = (cov_cond + cov_cond.T) / 2 + eps * np.eye(2)

            # Sample the costs
            y = self._rng.multivariate_normal(mu_cond, cov_cond)

            # Store the costs
            c_fix[i] = np.exp(y[0])
            c_var[i] = np.exp(y[1])

        # Clip the costs
        c_fix = np.clip(c_fix, clip["clip_lo"], clip["clip_hi_fix"])
        c_var = np.clip(c_var, clip["clip_lo"], clip["clip_hi_var"])

        return c_fix, c_var

    @staticmethod
    def _load_model(path: Path):
        """
        Loads a pre-fitted model from the given path.
        
        Args:
            path (Path): Path to the pre-fitted model.
            
        Returns:
            model (dict): Pre-fitted model.
        """

        # Load the model from the given path
        with open(path, "rb") as f:
            model =pickle.load(f)

        return model

# =============================================================================
# Data Generator (orchestrator)
# =============================================================================

class DataGenerator:
    """
    Implements an orchestrator for weight, distance, and cost generation for synthetic experiment runs.

    Receives a ``SeedManager`` (experiment-level) and obtains independent RNGs
    for each sub-generator via ``seed_manager.get_rng(name)``.

    Attributes:
        _raw_data_path (Path): Path to the raw data.
        _models_path (Path): Path to the pre-fitted models.
        _seed_manager (Optional[SeedManager]): Experiment-level seed manager.
    """

    def __init__(self, raw_data_path: str, models_path: str, seed_manager: Optional['SeedManager'] = None):
        """
        Args:
            raw_data_path (Path): Path to the raw CSV data directory.
            models_path (Path): Path to the directory containing pre-fitted ``.pkl`` models.
            seed_manager (Optional['SeedManager']): Experiment-level ``SeedManager`` instance (with
                ``EXPERIMENT_SEEDS`` registry).  ``None`` → unseeded generators.
        """
        self._raw_data_path = Path(raw_data_path)
        self._models_path = Path(models_path)
        self._seed_manager = seed_manager

    def generate(self, n_warehouses: int, n_skus: int, n_regions: int) -> Dict[str, Any]:
        """
        Generates the full synthetic data generation pipeline.

        Args:
            n_warehouses (int): Number of warehouses.
            n_skus (int): Number of SKUs.
            n_regions (int): Number of regions.

        Returns:
            results (Dict[str, Any]): Dictionary containing the generated data.
        """

        # Generate the weights
        sku_weights = self._generate_weights(n_skus)

        # Generate the distances
        location_data = self._generate_location_data(n_regions, n_warehouses, n_skus)

        # Generate the costs
        costs = self._generate_costs(location_data)

        # Generate the penalty cost
        penalty_cost = self._generate_penalty_cost(n_skus)

        # Package the results
        results = self._package_results(sku_weights, location_data, costs, penalty_cost)

        return results

    def _generate_weights(self, n_skus: int) -> np.ndarray:
        """
        Generates synthetic weights.

        Args:
            n_skus (int): Number of SKUs to generate weights for.

        Returns:
            sku_weights (np.ndarray): Synthetic SKU weights.
        """

        # Get random number generator
        rng = self._seed_manager.get_rng('data_weights') if self._seed_manager else np.random.default_rng()

        # Initialize WeightGenerator instance
        gen = WeightGenerator(
            model_path=self._models_path / "weight_gmm.pkl",
            rng=rng,
        )

        # Generate synthetic weights
        sku_weights = gen.generate(n_skus)

        return sku_weights

    def _generate_location_data(self, n_regions: int, n_warehouses: int, n_skus: int) -> dict:
        """
        Generates synthetic distances.
        
        Args:
            n_regions (int): Number of regions to generate distances for.
            n_warehouses (int): Number of warehouses to generate distances for.
            n_skus (int): Number of SKUs to generate distances for.

        Returns:
            distances (dict): Dictionary containing the generated distances.
        """

        # Get random number generator
        rng = self._seed_manager.get_rng('data_distances') if self._seed_manager else np.random.default_rng()

        # Initialize LocationDataGenerator  instance
        gen = LocationDataGenerator (
            model_path=self._models_path / "distance_gmm_geo.pkl",
            raw_data_path=self._raw_data_path,
            rng=rng,
        )

        # Generate the distances
        location_data = gen.generate(n_regions, n_warehouses, n_skus)

        return location_data

    def _generate_costs(self, distances: dict) -> dict:
        """
        Generates synthetic costs.

        Args:
            distances (dict): Dictionary containing the generated distances.

        Returns:
            costs (dict): Dictionary containing the generated costs.
        """

        # Get random number generator
        rng = self._seed_manager.get_rng('data_costs') if self._seed_manager else np.random.default_rng()
        gen = CostGenerator(
            outbound_model_path=self._models_path / "costs_gmm_outbound.pkl",
            inbound_model_path=self._models_path / "costs_gmm_inbound.pkl",
            raw_data_path=self._raw_data_path,
            rng=rng,
        )

        # Generate the costs
        costs = gen.generate(distances["outbound_distances"], distances["inbound_distances"])

        return costs

    @staticmethod
    def _generate_penalty_cost(n_skus: int) -> np.ndarray:
        return _bounded_increasing(PENALTY_MIN, PENALTY_MAX, n_skus, amp=0.02, seed=1)

    @staticmethod
    def _package_results(
        sku_weights: np.ndarray, location_data: dict, costs: dict, penalty_cost: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Assembles final dict with rounded nested lists for config compatibility.
        
        Args:
            sku_weights (np.ndarray): Synthetic SKU weights.
            location_data (dict): Dictionary containing the generated location data.
            costs (dict): Dictionary containing the generated costs.
            penalty_cost (np.ndarray): Synthetic penalty cost.

        Returns:
            results (Dict[str, Any]): Dictionary containing the generated results.
        """

        # Package the results
        results = {
            "penalty_cost": [round(float(x), 3) for x in penalty_cost],
            "sku_weights": sku_weights.round(3).tolist(),
            "max_wh_capacities": location_data["max_wh_capacities"].round(3).tolist(),
            "distances": location_data["outbound_distances"].round(3).tolist(),
            "outbound_fixed": costs["outbound_fixed"].round(3).tolist(),
            "outbound_variable": costs["outbound_variable"].round(5).tolist(),
            "inbound_fixed": costs["inbound_fixed"].round(3).tolist(),
            "inbound_variable": costs["inbound_variable"].round(5).tolist(),
            "lead_times": [[int(x) for x in row] for row in location_data["lead_times"]],
        }

        return results
