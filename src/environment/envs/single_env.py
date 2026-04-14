"""
Single-agent Gymnasium wrapper around :class:`InventoryEnvironment`.

``CentralizedEnvWrapper`` exposes the multi-warehouse inventory problem as a
standard single-agent ``gymnasium.Env``.  A single policy receives the full
global observation (concatenation of every warehouse's local features) and
outputs a joint action vector that is split and dispatched to individual
warehouses inside the wrapped PettingZoo environment.

This wrapper is used by the **PPO-Centralized** baseline, which removes all
MARL challenges (non-stationarity, partial observability, credit assignment)
and serves as an upper bound on what RL can achieve in this environment.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium.spaces import Box

from src.config.schema import EnvironmentConfig
from src.environment.envs.multi_env import InventoryEnvironment


class CentralizedEnvWrapper(gymnasium.Env):
    """
    Wraps :class:`InventoryEnvironment` (PettingZoo ``ParallelEnv``) as a
    single-agent ``gymnasium.Env``.

    Observation
        The global observation: concatenation of all warehouses' local feature
        vectors.  Shape: ``(n_warehouses * local_obs_dim,)``.

    Action
        Flat continuous vector in ``[-1, 1]`` covering all warehouses' SKUs.
        Shape: ``(n_warehouses * n_skus,)``.  Internally split into per-warehouse
        sub-actions and forwarded to the underlying environment.

    Reward
        Sum of per-warehouse rewards (team-scope scalar).

    Seeding
        All randomness is managed by the inner ``InventoryEnvironment``'s
        ``SeedManager``.  The wrapper does not create any additional RNG state.
    """

    metadata = {"render_modes": ["human"], "name": "single_env"}

    def __init__(
        self,
        env_config: EnvironmentConfig,
        seed: Optional[int] = None,
        env_meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            env_config: Validated environment configuration.
            seed: Root seed forwarded to the inner ``InventoryEnvironment``.
            env_meta: RLlib metadata dict (``data_mode``, ``obs_normalization``,
                ``obs_stats``, ``num_eval_episodes``, …) forwarded as-is.
        """
        super().__init__()

        self.env = InventoryEnvironment(env_config, seed=seed, env_meta=env_meta)

        self.env_config = env_config
        self.n_warehouses = self.env.n_warehouses
        self.n_skus = self.env.n_skus

        local_obs_dim = self.env._compute_local_obs_dim()
        self._local_obs_dim = local_obs_dim
        self._global_obs_dim = self.n_warehouses * local_obs_dim

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._global_obs_dim,),
            dtype=np.float32,
        )

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_warehouses * self.n_skus,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the inner environment and returns the global observation."""
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        global_obs = self._extract_global_obs(obs_dict)
        info = info_dict.get(self.env.agents[0], {})
        return global_obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Splits the joint action, steps the inner environment, and
        aggregates observations / rewards into single-agent format."""

        per_wh_actions = self._split_action(action)

        obs_dict, rewards, terminations, truncations, info_dict = self.env.step(
            per_wh_actions
        )

        global_obs = self._extract_global_obs(obs_dict)
        total_reward = sum(rewards.values())
        terminated = all(terminations.values())
        truncated = all(truncations.values())
        info = info_dict.get(self.env.agents[0], {})

        return global_obs, total_reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Properties forwarded from the inner environment
    # ------------------------------------------------------------------

    @property
    def collect_step_info(self) -> bool:
        return self.env.collect_step_info

    @collect_step_info.setter
    def collect_step_info(self, value: bool):
        self.env.collect_step_info = value

    @property
    def agents(self):
        """Warehouse agent IDs from the inner environment (for rollout code)."""
        return self.env.agents

    @property
    def episode_length(self) -> int:
        return self.env.episode_length

    @property
    def max_expected_lead_time(self) -> int:
        return self.env.max_expected_lead_time

    @property
    def feature_config(self):
        return self.env.feature_config

    @property
    def include_warehouse_id(self) -> bool:
        return self.env.include_warehouse_id

    @property
    def rolling_window(self) -> int:
        return self.env.rolling_window

    @property
    def obs_normalization(self):
        return self.env.obs_normalization

    @obs_normalization.setter
    def obs_normalization(self, value):
        self.env.obs_normalization = value

    @property
    def obs_stats(self):
        return self.env.obs_stats

    @obs_stats.setter
    def obs_stats(self, value):
        self.env.obs_stats = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_global_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Extracts the global observation from the per-agent observations.

        Each agent's observation is ``concat(local_i, global)`` where
        ``global = concat(local_0, local_1, …)``.  The global portion is
        identical across agents, so we extract it from the first agent.
        """
        first_obs = obs_dict[self.env.agents[0]]
        global_obs = first_obs[self._local_obs_dim:]
        return global_obs

    def _split_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Splits a flat ``(W*K,)`` action into per-warehouse sub-actions."""
        K = self.n_skus
        return {
            agent_id: action[i * K : (i + 1) * K]
            for i, agent_id in enumerate(self.env.agents)
        }
