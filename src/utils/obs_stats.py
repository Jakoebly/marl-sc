from typing import TYPE_CHECKING, Tuple

import numpy as np

from src.config.schema import EnvironmentConfig, FeatureConfig

if TYPE_CHECKING:
    from src.utils.seed_manager import SeedManager


def compute_obs_statistics(
    env_config: EnvironmentConfig,
    seed_manager: 'SeedManager',
    mode: str = "meanstd_custom",
    n_episodes: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-feature mean and std by running a random policy for
    ``n_episodes``. Supports three modes:

    - ``meanstd_custom``: pooled across warehouses; each feature dimension
      gets its own (mean, std).
    - ``meanstd_sku_grouped``: pooled across warehouses; per-SKU dimensions
      within each feature group share a single (mean, std); aggregate
      dimensions get their own independent (mean, std).
    - ``meanstd_wh_grouped``: per-warehouse statistics
      with the same SKU-grouping rule as ``meanstd_sku_grouped`` applied
      independently inside each warehouse's bucket.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        seed_manager (SeedManager): Experiment-level seed manager. Must have
            ``'obs_stats'`` registered (it is part of ``EXPERIMENT_SEEDS``).
        mode (str): ``"meanstd_custom"``, ``"meanstd_sku_grouped"``, or
            ``"meanstd_wh_grouped"``.
        n_episodes (int): Number of episodes to collect.

    Returns:
        (obs_mean, obs_std) (Tuple[np.ndarray, np.ndarray]): Tuple containing
            the mean and standard deviation of the observations. Shape is
            ``(local_obs_dim,)`` for the pooled modes and
            ``(n_warehouses, local_obs_dim)`` for ``meanstd_wh_grouped``.
    """

    from src.environment.envs.multi_env import InventoryEnvironment

    # Spawn two independent child seeds from the 'obs_stats' slot
    env_seed, action_seed = seed_manager.spawn_child_seeds("obs_stats", 2)

    # Initialize environment with the env seed
    env = InventoryEnvironment(env_config, seed=env_seed)
    n_skus = env.n_skus
    local_obs_dim = env._compute_local_obs_dim()

    # Initialize action RNG with the action seed
    action_rng = np.random.default_rng(action_seed)

    # Run random policy for n_episodes and collect local observations
    per_agent_obs = {agent_id: [] for agent_id in env.agents}
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            for agent_id in env.agents:
                per_agent_obs[agent_id].append(obs[agent_id][:local_obs_dim])
            actions = {
                agent_id: action_rng.uniform(-1, 1, size=(n_skus,)).astype(np.float32)
                for agent_id in env.agents
            }
            obs, _, terms, truncs, _ = env.step(actions)
            done = all(truncs.values()) or all(terms.values())
        for agent_id in env.agents:
            per_agent_obs[agent_id].append(obs[agent_id][:local_obs_dim])

    per_agent_arr = {
        agent_id: np.array(samples, dtype=np.float32)
        for agent_id, samples in per_agent_obs.items()
    }

    # Compute mean and std for the observations based on the mode
    if mode == "meanstd_wh_grouped":
        means, stds = [], []
        for agent_id in env.agents:
            m, s = _compute_grouped_stats(
                per_agent_arr[agent_id], n_skus, env.max_expected_lead_time,
                env.rolling_window, env.feature_config,
            )
            means.append(m)
            stds.append(s)
        obs_mean = np.stack(means, axis=0)
        obs_std = np.stack(stds, axis=0)
    else:
        all_local_obs = np.concatenate(
            [per_agent_arr[agent_id] for agent_id in env.agents], axis=0,
        )
        if mode == "meanstd_sku_grouped":
            obs_mean, obs_std = _compute_grouped_stats(
                all_local_obs, n_skus, env.max_expected_lead_time,
                env.rolling_window, env.feature_config,
            )
        else:
            obs_mean = all_local_obs.mean(axis=0)
            obs_std = all_local_obs.std(axis=0)
            if env.feature_config.sku_weight:
                weight_block = all_local_obs[:, -n_skus:]
                obs_mean[-n_skus:] = weight_block.mean()
                obs_std[-n_skus:] = weight_block.std()

    obs_std = np.where(obs_std < 1e-8, 1.0, obs_std)

    n_samples = sum(arr.shape[0] for arr in per_agent_arr.values())
    print(
        f"[INFO] Computed obs statistics ({mode}) from {n_episodes} episodes "
        f"({n_samples} samples, feature_dim={local_obs_dim})"
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
    if features.sku_weight:
        groups.append((n_skus, False))

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

    # Guard against silent drift from _build_local_obs
    assert idx == feature_dim, (
        f"grouped-stats layout covers {idx} columns but the observation "
        f"dimension is {feature_dim}!"
    )

    return obs_mean, obs_std
