"""Utilities for exporting and loading RLModule weights across training runs.

Used by the curriculum learning pipeline to warm-start a new training run
from the weights of a previously trained policy.  Shape mismatches are
handled automatically: parameters whose shapes match are transferred while
incompatible parameters are kept at their (random) initialisation.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def export_module_weights(trainer: Any, policy_id: str, output_path: str) -> None:
    """
    Saves an RLModule's ``state_dict`` to a ``.pt`` file.

    Args:
        trainer (Any): Built RLlib Algorithm instance (``PPO``, etc.).
        policy_id (str): Module / policy identifier (e.g. ``"shared_policy"``).
        output_path (str): Destination file path (should end in ``.pt``).
    """

    # Get the module from the trainer
    module = trainer.get_module(module_id=policy_id)

    # Create the parent directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the state dict to the output path and print confirmation
    torch.save(module.state_dict(), output_path)
    print(f"[WARMSTART] Exported weights for '{policy_id}' → {output_path}")


def load_module_weights(
    trainer: Any,
    policy_id: str,
    weights_path: str,
) -> Dict[str, List[str]]:
    """
    Loads weights into an RLModule from a ``.pt`` file.

    Automatically compares parameter shapes between the source file and the
    target module.  Matching parameters are copied; mismatched or missing
    parameters are left at their current (random) values and logged.

    After loading, the caller is responsible for syncing the updated weights
    to env runners (``trainer.learner_group.get_weights()`` →
    ``trainer.env_runner.set_weights(...)``).

    Args:
        trainer: Built RLlib Algorithm instance.
        policy_id: Module / policy identifier.
        weights_path: Path to a ``.pt`` file produced by
            :func:`export_module_weights`.

    Returns:
        loaded_dict (Dict[str, List[str]]): Dictionary with ``"loaded"`` and ``"skipped"`` keys.  ``"loaded"``
        lists the parameter names that were successfully transferred.
        ``"skipped"`` lists ``(name, reason)`` tuples for parameters that
        could not be transferred.
    """

    # Get the module from the trainer 
    module = trainer.get_module(module_id=policy_id)

    # Get the source state from the file (i.e., the weights with dimensions from the previous training run)
    source_state: Dict[str, torch.Tensor] = torch.load(
        weights_path, map_location="cpu", weights_only=True,
    )

    # Get the target state from the module (i.e., the weights with dimensions from the current training run)
    target_state = module.state_dict()

    # Initialize lists to store loaded and skipped parameters
    loaded: List[str] = []
    skipped: List[Tuple[str, str]] = []

    # Iterate over the source state and compare the shapes of the parameters
    for key, param in source_state.items():

        # If the parameter is not in the target state, add it to the skipped list
        if key not in target_state: 
            skipped.append((key, "missing in target"))
        # If the parameter is in the target state but has a different shape, add it to the skipped list
        elif param.shape != target_state[key].shape:
            skipped.append(
                (key, f"shape mismatch: {list(param.shape)} vs {list(target_state[key].shape)}")
            )
        # If the parameter is in the target state and has the same shape, copy the parameter to the target state
        else:
            target_state[key] = param
            loaded.append(key)

    # Load the target state into the module
    module.load_state_dict(target_state)

    # Print results
    if skipped:
        print(
            f"[WARMSTART] Partial load for '{policy_id}': "
            f"{len(loaded)} params loaded, {len(skipped)} skipped"
        )
        for key, reason in skipped:
            print(f"  - {key}: {reason}")
    else:
        print(
            f"[WARMSTART] Full load for '{policy_id}': "
            f"all {len(loaded)} params loaded from {weights_path}"
        )

    # Return the loaded and skipped parameters
    loaded_dict = {"loaded": loaded, "skipped": skipped}
    
    return loaded_dict
