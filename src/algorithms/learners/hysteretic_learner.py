from typing import Any, Dict

from ray.rllib.algorithms.ppo import PPOConfig as RLlibPPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModuleID, TensorType

torch, _ = try_import_torch()


class HystereticPPOTorchLearner(PPOTorchLearner):
    """PPO learner with asymmetric advantage weighting for MARL stability.

    Downweights negative advantages by a factor ``hysteretic_beta`` so that
    the policy update trusts "this action was good" signals more than "this
    action was bad" signals.  This reduces the impact of poor outcomes caused
    by other agents' exploration, stabilizing independent-learning MARL.

    The beta value is read from ``config.learner_config_dict["hysteretic_beta"]``
    at each loss computation.  A value of 1.0 recovers standard PPO.
    """

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: RLlibPPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:

        # Get beta value from config
        beta = config.learner_config_dict.get("hysteretic_beta", 1.0)

        # Apply hysteretic weighting if beta < 1.0
        if beta < 1.0:
            adv = batch[Postprocessing.ADVANTAGES]
            weights = torch.where(adv >= 0, 1.0, beta)
            batch[Postprocessing.ADVANTAGES] = adv * weights

        # Compute loss
        loss = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )

        return loss
