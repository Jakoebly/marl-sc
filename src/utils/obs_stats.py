from typing import Optional, Tuple

import numpy as np

from src.config.schema import EnvironmentConfig, FeatureConfig


def compute_obs_statistics(
    env_config: EnvironmentConfig,
    mode: str = "meanstd_custom",
    n_episodes: int = 10,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-feature mean and std by running a random policy for
    ``n_episodes``. Supports two modes:

    - ``meanstd_custom``: each feature dimension gets its own (mean, std).
    - ``meanstd_grouped``: per-SKU dimensions within each feature group share
      a single (mean, std); aggregate dimensions get their own independent
      (mean, std).

    The returned arrays have shape ``(local_obs_dim,)`` where the dimension
    is computed dynamically from the environment's feature config.
    Features with near-zero std are set to 1.0.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        mode (str): ``"meanstd_custom"`` or ``"meanstd_grouped"``.
        n_episodes (int): Number of episodes to collect.
        seed (Optional[int]): Seed for reproducibility.

    Returns:
        (obs_mean, obs_std) (Tuple[np.ndarray, np.ndarray]): Tuple containing the mean and standard deviation of the observations. 
            Shape of each np.ndarray: (local_obs_dim,).
    """

    from src.environment.envs.multi_env import InventoryEnvironment

    env = InventoryEnvironment(env_config, seed=seed)
    n_skus = env.n_skus
    local_obs_dim = env._compute_local_obs_dim()

    action_rng = np.random.default_rng(seed)
    all_local_obs = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            for agent_id in env.agents:
                all_local_obs.append(obs[agent_id][:local_obs_dim])

            actions = {
                agent_id: action_rng.uniform(-1, 1, size=(n_skus,)).astype(np.float32)
                for agent_id in env.agents
            }
            obs, _, terms, truncs, _ = env.step(actions)
            done = all(truncs.values()) or all(terms.values())

        for agent_id in env.agents:
            all_local_obs.append(obs[agent_id][:local_obs_dim])

    all_local_obs = np.array(all_local_obs, dtype=np.float32)

    if mode == "meanstd_grouped":
        obs_mean, obs_std = _compute_grouped_stats(
            all_local_obs, n_skus, env.max_expected_lead_time,
            env.rolling_window, env.feature_config,
        )
    else:
        obs_mean = all_local_obs.mean(axis=0)
        obs_std = all_local_obs.std(axis=0)

    obs_std = np.where(obs_std < 1e-8, 1.0, obs_std)

    print(
        f"[INFO] Computed obs statistics ({mode}) from {n_episodes} episodes "
        f"({len(all_local_obs)} samples, feature_dim={local_obs_dim})"
    )

    return obs_mean, obs_std


def _compute_grouped_stats(
    all_obs: np.ndarray,
    n_skus: int,
    max_expected_lead_time: int,
    rolling_window: int,
    feature_config: FeatureConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes shared (mean, std) for per-SKU columns within each feature group.
    Aggregate columns get their own independent (mean, std).  Only enabled
    features (as determined by *feature_config*) are included.

    Args:
        all_obs (np.ndarray): Collected observations. Shape: (N, local_obs_dim).
        n_skus (int): Number of SKUs.
        max_expected_lead_time (int): Maximum expected lead time (pipeline slots).
        rolling_window (int): Rolling window size (demand history slots).
        feature_config (FeatureConfig): Active feature configuration.

    Returns:
        (obs_mean, obs_std) (Tuple[np.ndarray, np.ndarray]): Tuple containing
            the mean and standard deviation of the observations.
            Shape of each np.ndarray: (local_obs_dim,).
    """

    # Get feature config
    features = feature_config

    # Build groups list dynamically: (sku_column_count, has_aggregate)
    groups = []
    if features.inventory:
        groups.append((n_skus, features.inventory_aggregate))
    if features.pipeline:
        groups.append((max_expected_lead_time * n_skus, features.pipeline_aggregate))
    if features.incoming_demand_home:
        groups.append((n_skus, features.incoming_demand_home_aggregate))
    if features.units_shipped_home:
        groups.append((n_skus, False))
    if features.units_shipped_away:
        groups.append((n_skus, features.units_shipped_away_aggregate))
    if features.stockout:
        groups.append((n_skus, False))
    if features.rolling_demand_mean:
        groups.append((n_skus, features.rolling_demand_mean_aggregate))
    if features.demand_forecast:
        groups.append((n_skus, features.demand_forecast_aggregate))
    if features.days_of_supply:
        groups.append((n_skus, False))
    if features.net_inventory_position:
        groups.append((n_skus, False))
    if features.demand_variability:
        groups.append((n_skus, False))
    if features.demand_history:
        groups.append((rolling_window * n_skus, False))

    # Initialize mean and std arrays
    feature_dim = all_obs.shape[1]
    obs_mean = np.zeros(feature_dim, dtype=np.float32)
    obs_std = np.ones(feature_dim, dtype=np.float32)

    # Compute mean and std for each feature group
    idx = 0
    for sku_count, has_agg in groups:
        sku_data = all_obs[:, idx:idx + sku_count]
        shared_mean = float(sku_data.mean())
        shared_std = float(sku_data.std())

        obs_mean[idx:idx + sku_count] = shared_mean
        obs_std[idx:idx + sku_count] = shared_std
        idx += sku_count

        if has_agg:
            agg_col = all_obs[:, idx]
            obs_mean[idx] = float(agg_col.mean())
            obs_std[idx] = float(agg_col.std())
            idx += 1

    return obs_mean, obs_std
