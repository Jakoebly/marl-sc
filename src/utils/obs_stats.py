from typing import Optional, Tuple

import numpy as np

from src.config.schema import EnvironmentConfig


def compute_obs_statistics(
    env_config: EnvironmentConfig,
    n_episodes: int = 10,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-feature mean and std of local observations by running a random
    policy in the environment for ``n_episodes``.

    The returned arrays have shape ``(local_obs_dim,)`` where
    ``local_obs_dim = 8 * n_skus + 6``. Features with near-zero std are set
    to 1.0 to avoid division-by-zero during normalization.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        n_episodes (int): Number of episodes to collect. Defaults to 10.
        seed (Optional[int]): Seed for reproducibility (env stochastics and
            action sampling). Defaults to None.

    Returns:
        Tuple of (obs_mean, obs_std), each np.ndarray of shape (local_obs_dim,).
    """

    from src.environment.envs.multi_env import InventoryEnvironment

    env = InventoryEnvironment(env_config, seed=seed)
    local_obs_dim = 8 * env.n_skus + 6

    action_rng = np.random.default_rng(seed)
    all_local_obs = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            for agent_id in env.agents:
                all_local_obs.append(obs[agent_id][:local_obs_dim])

            actions = {
                agent_id: action_rng.uniform(-1, 1, size=(env.n_skus,)).astype(np.float32)
                for agent_id in env.agents
            }
            obs, _, terms, truncs, _ = env.step(actions)
            done = all(truncs.values()) or all(terms.values())

        for agent_id in env.agents:
            all_local_obs.append(obs[agent_id][:local_obs_dim])

    all_local_obs = np.array(all_local_obs, dtype=np.float32)
    obs_mean = all_local_obs.mean(axis=0)
    obs_std = all_local_obs.std(axis=0)
    obs_std = np.where(obs_std < 1e-8, 1.0, obs_std)

    print(
        f"[INFO] Computed obs statistics from {n_episodes} episodes "
        f"({len(all_local_obs)} samples, local_obs_dim={local_obs_dim})"
    )

    return obs_mean, obs_std
