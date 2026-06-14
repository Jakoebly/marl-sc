"""Per-SKU reward decomposition learner.

This module implements the learner side of the per-SKU reward decomposition. It pairs a
custom GAE learner-connector that computes one advantage and value target per SKU channel 
with a PPO learner that rewrites the policy surrogate into a per-SKU factored form. 
With a single value head and the pooled reward both collapse exactly to standard PPO, so 
per-SKU reward decomposition is a strict generalization switched on by the per-SKU 
reward channels.

The implementation mirrors RLlib's ``GeneralAdvantageEstimation`` connector and
``PPOTorchLearner`` (Ray 2.52) and changes only what the decomposition requires:
- per-SKU rewards are read from the episode infos and GAE is run per channel by 
  reusing RLlib's ``compute_value_targets``;
- the critic produces one value per SKU and the value loss is summed over channels;
- the diagonal-Gaussian log-probability ratio is kept per action coordinate (i.e. per SKU)
  instead of being summed, so each SKU coordinate is weighted by its own advantage.
"""

from typing import Any, Dict, List

import numpy as np

from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
    PPOConfig,
)
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import ENTROPY_KEY, POLICY_LOSS_KEY, VF_LOSS_KEY
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.postprocessing.value_predictions import compute_value_targets
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.typing import EpisodeType, ModuleID, TensorType

from src.environment.components.reward_calculator import REWARD_PER_SKU_INFO_KEY

torch, nn = try_import_torch()


class PerSkuGeneralAdvantageEstimation(ConnectorV2):
    """GAE learner-connector variant that computes one advantage/value target per SKU.

    Mirrors RLlib's ``GeneralAdvantageEstimation`` but treats the value function as a
    vector of per-SKU heads. Per-SKU rewards are read from the episode infos and GAE is
    run independently per channel by reusing RLlib's verified ``compute_value_targets``
    (which is not vectorizable over a trailing channel dimension because it appends a
    scalar bootstrap to the value array). Advantages are standardized over all channels
    jointly, which reduces to RLlib's behavior when there is a single channel.
    """

    def __init__(
        self,
        input_observation_space=None,
        input_action_space=None,
        *,
        gamma,
        lambda_,
    ):
        """Initializes a PerSkuGeneralAdvantageEstimation instance.

        Args:
            input_observation_space(Space): The observation space
            input_action_space(Space): The action space
            gamma(float): The discount factor gamma
            lambda_(float): The lambda parameter for Per-SKU GAE
        """

        # Initialize the base class and set the gamma and lambda parameters
        super().__init__(input_observation_space, input_action_space)
        self.gamma = gamma
        self.lambda_ = lambda_

        # Initialize the internal numpy-to-tensor connector to translate 
        # the GAE results to tensors
        self._numpy_to_tensor_connector = None

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: MultiRLModule,
        episodes: List[EpisodeType],
        batch: Dict[str, Any],
        **kwargs,
    ):
        """Computes the GAE results for each module.

        Args:
            rl_module(MultiRLModule): The given RL module
            episodes(List[EpisodeType]): The episodes data
            batch(Dict[str, Any]): The batch data
        """

        # Set the device to place all GAE result tensors on
        device = None

        # Extract all single-agent episodes
        sa_episodes_list = list(
            self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False)
        )

        # Get the per-SKU value-function forward passes for every value-based module
        vf_preds = rl_module.foreach_module(
            func=lambda mid, module: (
                module.compute_values(batch[mid])
                if mid in batch and isinstance(module, ValueFunctionAPI)
                else None
            ),
            return_dict=True,
        )

        # Loop through all modulesand compute the GAE results for each
        for module_id, module_vf_preds in vf_preds.items():
            # Skip modules that are not value-function implementers
            if module_vf_preds is None:
                continue
            
            # Get the module
            module = rl_module[module_id]
            
            # Safeguard against stateful (GRU) modules 
            assert not module.is_stateful(), (
                "reward_decomposition does not support stateful (GRU) modules."
            )

            # Convert the module value function predictions to numpy
            device = module_vf_preds.device
            module_vf_preds = convert_to_numpy(module_vf_preds)

            # Check if the module value function predictions have the correct shape
            if module_vf_preds.ndim != 2:
                raise ValueError(
                    "PerSkuGeneralAdvantageEstimation expects a per-SKU value head of "
                    f"shape (T, n_skus); got {module_vf_preds.shape}. Is "
                    "model_config['value_head_dim'] set to n_skus?"
                )
            
            # Get the number of SKUs from shape
            num_sku = module_vf_preds.shape[1]

            # Collect (single-agent) episode lengths for this particular module
            episode_lens = [
                len(e) for e in sa_episodes_list if e.module_id in [None, module_id]
            ]

            # Get the terminateds, truncateds and scalar rewards from the batch
            terminateds = convert_to_numpy(batch[module_id][Columns.TERMINATEDS])
            truncateds = convert_to_numpy(batch[module_id][Columns.TRUNCATEDS])
            scalar_rewards = convert_to_numpy(batch[module_id][Columns.REWARDS])

            # Check if per-SKU rewards aligns with `scalar_rewards`
            rewards_per_sku = self._rewards_per_sku_from_episodes(
                sa_episodes_list, module_id, num_sku
            )
            assert rewards_per_sku.shape == module_vf_preds.shape, (
                f"per-SKU reward shape {rewards_per_sku.shape} does not match value "
                f"prediction shape {module_vf_preds.shape}."
            )
            assert np.allclose(rewards_per_sku.sum(axis=1), scalar_rewards, atol=1e-4), (
                "Per-SKU rewards read from infos do not sum to Columns.REWARDS; the env "
                "decomposition or the info alignment is inconsistent."
            )

            # Compute value targets per SKU channel via RLlib's GAE math
            module_value_targets = np.stack(
                [
                    compute_value_targets(
                        values=module_vf_preds[:, s],
                        rewards=rewards_per_sku[:, s],
                        terminateds=terminateds,
                        truncateds=truncateds,
                        gamma=self.gamma,
                        lambda_=self.lambda_,
                    )
                    for s in range(num_sku)
                ],
                axis=1,
            )
            assert module_value_targets.shape[0] == sum(episode_lens)

    	    # Compute advantages
            module_advantages = module_value_targets - module_vf_preds

            # Standardize advantages jointly over all timesteps and channels
            module_advantages = (module_advantages - module_advantages.mean()) / max(
                1e-4, module_advantages.std()
            )

            # Update the batch with the advantages and value targets
            batch[module_id][Postprocessing.ADVANTAGES] = module_advantages
            batch[module_id][Postprocessing.VALUE_TARGETS] = module_value_targets

        # Convert all GAE results to tensors
        if self._numpy_to_tensor_connector is None:
            self._numpy_to_tensor_connector = NumpyToTensor(
                as_learner_connector=True, device=device
            )
        tensor_results = self._numpy_to_tensor_connector(
            rl_module=rl_module,
            batch={
                mid: {
                    Postprocessing.ADVANTAGES: module_batch[Postprocessing.ADVANTAGES],
                    Postprocessing.VALUE_TARGETS: (
                        module_batch[Postprocessing.VALUE_TARGETS]
                    ),
                }
                for mid, module_batch in batch.items()
                if vf_preds.get(mid) is not None
            },
            episodes=episodes,
        )

        # Move converted tensors back to `batch`.
        for mid, module_batch in tensor_results.items():
            batch[mid].update(module_batch)

        return batch

    @staticmethod
    def _rewards_per_sku_from_episodes(
        sa_episodes_list: List[EpisodeType],
        module_id: ModuleID,
        num_sku: int,
    ) -> np.ndarray:
        """Builds the per-SKU reward array aligned with ``Columns.REWARDS``.

        Episode infos are aligned with observations (index 0 is the reset info), so the
        per-SKU reward produced at env step ``t`` lives at info index ``t + 1``. Reading
        indices ``1 .. ep_len - 1`` therefore yields the real per-step rewards in order;
        the artificial bootstrap timestep appended by ``AddOneTsToEpisodesAndTruncate``
        gets a zero row, exactly like the 0.0 scalar reward appended there.
        """
        # Initialize the list of per-SKU rewards and the zero row
        rows: List[np.ndarray] = []
        zero_row = np.zeros(num_sku, dtype=np.float32)

        # Loop through all single-agent episodes and build the per-SKU reward array
        for sa_episode in sa_episodes_list:
            if sa_episode.module_id not in [None, module_id]:
                continue
            ep_len = len(sa_episode)  
            step_infos = sa_episode.get_infos(slice(1, ep_len))
            for info in step_infos:
                rps = info.get(REWARD_PER_SKU_INFO_KEY) if isinstance(info, dict) else None
                rows.append(
                    np.asarray(rps, dtype=np.float32) if rps is not None else zero_row
                )
            rows.append(zero_row)  

        return np.stack(rows, axis=0)


class RewardDecompositionPPOTorchLearner(PPOTorchLearner):
    """PPO learner with a per-SKU factored surrogate and per-SKU value loss.

    Swaps RLlib's default scalar GAE learner-connector for the per-SKU variant and
    rewrites the PPO loss so that each SKU's clipped surrogate is weighted by its own
    advantage. The diagonal-Gaussian factorization makes this exact: the joint
    log-probability is the sum of per-coordinate log-probabilities, so keeping the
    per-coordinate ratios recovers standard PPO once they are summed back.
    """

    @override(PPOLearner)
    def build(self) -> None:
        """Builds the RewardDecompositionPPOTorchLearner."""

        # Build the base PPO learner
        super().build()

        # Replace the scalar GAE connector added by PPOLearner.build with the per-SKU
        # variant
        if (
            self._learner_connector is not None
            and self.config.add_default_connectors_to_learner_pipeline
        ):
            self._learner_connector.remove("GeneralAdvantageEstimation")
            self._learner_connector.append(
                PerSkuGeneralAdvantageEstimation(
                    gamma=self.config.gamma, lambda_=self.config.lambda_
                )
            )

    @override(TorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:

        """Computes the loss for a given module.

        Args:
            module_id(ModuleID): The ID of the module
            config(PPOConfig): The configuration for the PPO learner
            batch(Dict[str, Any]): The train batch for the given module
            fwd_out(Dict[str, TensorType]): The output of the forward pass for the given module
        """

        # Get the module
        module = self.module[module_id].unwrapped()

        # If a mask is provided, mask out the artificial bootstrap timesteps
        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]
            num_valid = torch.sum(mask)

            def possibly_masked_mean(data_):
                return torch.sum(data_[mask]) / num_valid

        # If no mask is provided, use the mean
        else:
            possibly_masked_mean = torch.mean

        # Get the actions, advantages and value targets from the batch
        actions = batch[Columns.ACTIONS]  # Shape: (B, n_skus)
        advantages = batch[Postprocessing.ADVANTAGES]  # Shape: (B, n_skus)
        value_targets = batch[Postprocessing.VALUE_TARGETS]  # Shape: (B, n_skus)

        # Compute the per-SKU log-probability ratios
        new_loc, new_log_std = fwd_out[Columns.ACTION_DIST_INPUTS].chunk(2, dim=-1)
        old_loc, old_log_std = batch[Columns.ACTION_DIST_INPUTS].chunk(2, dim=-1)
        assert new_loc.shape == actions.shape, (
            "reward_decomposition assumes a diagonal Gaussian policy with one "
            f"action coordinate per SKU; got dist inputs {tuple(new_loc.shape)} for "
            f"actions {tuple(actions.shape)}."
        )
        new_logp = torch.distributions.Normal(new_loc, new_log_std.exp()).log_prob(actions)
        old_logp = torch.distributions.Normal(old_loc, old_log_std.exp()).log_prob(actions)
        logp_ratio = torch.exp(new_logp - old_logp)  # Shape: (B, n_skus)

        # Compute the factored surrogate loss as the sum of the per-SKU clipped 
        # objectives over the SKU dimension
        surrogate_loss = torch.sum(
            torch.min(
                advantages * logp_ratio,
                advantages
                * torch.clamp(logp_ratio, 1 - config.clip_param, 1 + config.clip_param),
            ),
            dim=-1,
        )  # Shape: (B,)

        # Get the entropy from the joint diagonal Gaussian
        action_dist_class_train = module.get_train_action_dist_cls()
        action_dist_class_exploration = module.get_exploration_action_dist_cls()
        curr_action_dist = action_dist_class_train.from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        curr_entropy = curr_action_dist.entropy()
        mean_entropy = possibly_masked_mean(curr_entropy)

        # Get the KL from the joint diagonal Gaussian if enabled
        if config.use_kl_loss:
            prev_action_dist = action_dist_class_exploration.from_logits(
                batch[Columns.ACTION_DIST_INPUTS]
            )
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = possibly_masked_mean(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=surrogate_loss.device)

        # Compute the per-SKU value loss as the sum over SKU channels of the squared errors
        if config.use_critic:
            value_fn_out = module.compute_values(
                batch, embeddings=fwd_out.get(Columns.EMBEDDINGS)
            )  # Shape: (B, n_skus)
            vf_loss = torch.pow(value_fn_out - value_targets, 2.0).sum(dim=-1)  # Shape: (B,)
            mean_vf_loss = possibly_masked_mean(vf_loss)
            mean_vf_unclipped_loss = mean_vf_loss
        else:
            z = torch.tensor(0.0, device=surrogate_loss.device)
            value_fn_out = mean_vf_unclipped_loss = vf_loss = mean_vf_loss = z

        # Compute the total loss
        total_loss = possibly_masked_mean(
            -surrogate_loss
            + config.vf_loss_coeff * vf_loss
            - (
                self.entropy_coeff_schedulers_per_module[module_id].get_current_value()
                * curr_entropy
            )
        )

        # Add the KL loss if enabled
        if config.use_kl_loss:
            total_loss += self.curr_kl_coeffs_per_module[module_id] * mean_kl_loss

        # Log important loss stats (flatten per-SKU tensors for explained variance).
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: -possibly_masked_mean(surrogate_loss),
                VF_LOSS_KEY: mean_vf_loss,
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    value_targets.reshape(-1), value_fn_out.reshape(-1)
                ),
                ENTROPY_KEY: mean_entropy,
                LEARNER_RESULTS_KL_KEY: mean_kl_loss,
            },
            key=module_id,
            window=1,
        )
        return total_loss
