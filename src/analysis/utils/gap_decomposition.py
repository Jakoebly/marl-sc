"""
Gap decomposition for phase 2 of the paper: where the gap to the central controller comes from.

Four variants run on every environment, each tuned and then seed-evaluated on the same seeds:

    shared_own     one policy shared by all warehouses, each warehouse rewarded on its own cost
    separate_own   one policy per warehouse, each warehouse rewarded on its own cost
    shared_team    one policy shared by all warehouses, every warehouse rewarded on the network cost
    central        one central controller (CPPO) for the whole network

Three effects are paired differences of network returns (higher return is better, so a positive
effect means the first variant beats the second):

    sharing    = R(separate_own) - R(shared_own)    cost of sharing one policy
    own_cost   = R(shared_team)  - R(shared_own)    cost of each warehouse caring only about its own bill
    local_view = R(central)      - R(shared_team)   cost of each warehouse seeing only its own stock

Each cell contributes its locked config: the config with the best mean in the cell's
``seed_evaluation_summary.yaml``. The summary stores summed episode returns, and a sum over the
warehouses counts the team part of the reward once per warehouse. Each value is therefore
divided by ``1 + team_weight * (W - 1)`` (W for the team reward, 1 for the agent reward), read from
the run's saved ``env_config.yaml``, before the cells are compared. The p-values of all effects
in the table are corrected together with Holm-Bonferroni.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Union

import yaml

from src.analysis.utils.holm_bonferroni import holm_bonferroni
from src.analysis.utils.paired_difference import paired_difference
from src.experiments.utils.experiment_utils import SEED_EVAL_SUBDIR, SEED_EVAL_SUMMARY_FILENAME

__all__ = [
    "VARIANTS",
    "EFFECTS",
    "PAPER_ENVS",
    "PAPER_EXPERIMENT_PREFIXES",
    "network_return_factor",
    "load_locked_cell",
    "paper_cells",
    "gap_decomposition",
]

VARIANTS = ("shared_own", "separate_own", "shared_team", "central")

# Effect name -> (variant A, variant B); the effect is R(A) - R(B)
EFFECTS = {
    "sharing": ("separate_own", "shared_own"),
    "own_cost": ("shared_team", "shared_own"),
    "local_view": ("central", "shared_team"),
}

PAPER_ENVS = ("Symmetric", "DemandHetero", "CostHetero", "SKUHetero", "CombinedHetero")

# Experiment name of a cell: f"{prefix}_{env}"
PAPER_EXPERIMENT_PREFIXES = {
    "shared_own": "P2_MAPPO_PS",
    "separate_own": "P2_MAPPO_noPS",
    "shared_team": "P2_MAPPO_PS_Team",
    "central": "P2_CPPO",
}

PathLike = Union[str, Path]


def network_return_factor(env_config_path: PathLike) -> float:
    """
    Returns how many times a summed episode return counts the network return under the reward
    of a saved environment config: ``1 + team_weight * (W - 1)``, i.e. 1 for the agent reward,
    W for the team reward and in between for the mixed reward.

    Args:
        env_config_path (PathLike): Saved ``env_config.yaml`` of a run.

    Returns:
        factor (float): Summed return divided by this factor is the network return.
    """

    # Read the reward scope and the number of warehouses
    data = yaml.safe_load(Path(env_config_path).read_text(encoding="utf-8"))
    env = data.get("environment", data)
    params = env["components"]["reward_calculator"]["params"]
    team_weight = {"agent": 0.0, "team": 1.0}.get(params["scope"], params.get("team_weight"))

    return 1.0 + float(team_weight) * (int(env["n_warehouses"]) - 1)

def load_locked_cell(tune_dir: PathLike) -> Dict[str, Any]:
    """
    Reads the locked config of one cell, i.e. the config with the best mean in the cell's
    ``seed_evaluation_summary.yaml``, and converts its per-seed returns to network returns.

    Args:
        tune_dir (PathLike): Tune experiment directory of the cell (contains ``seed_evaluation/``).

    Returns:
        cell (Dict[str, Any]): ``locked_config`` (name), ``factor`` (see
            :func:`network_return_factor`), ``mean`` and ``per_seed`` (``{seed: network return}``).
    """

    # Select the config with the best mean across seeds
    seed_dir = Path(tune_dir) / SEED_EVAL_SUBDIR
    summary = yaml.safe_load((seed_dir / SEED_EVAL_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    locked = max(summary["configs"], key=lambda config: config["mean"])

    # Read the reward of the locked config from one of its seed runs
    env_configs = sorted(seed_dir.glob(f"{locked['name']}/*/env_config.yaml"))
    if not env_configs:
        raise FileNotFoundError(f"No env_config.yaml in the seed runs of {seed_dir / locked['name']}")
    factor = network_return_factor(env_configs[0])

    # Convert the summed returns to network returns
    per_seed = {int(entry["seed"]): float(entry["value"]) / factor for entry in locked["per_seed"]}
    cell = {
        "locked_config": locked["name"],
        "factor": factor,
        "mean": sum(per_seed.values()) / len(per_seed),
        "per_seed": per_seed,
    }

    return cell

def paper_cells(
    root: PathLike = "experiment_outputs/Paper",
    envs: Iterable[str] = PAPER_ENVS,
) -> Dict[str, Dict[str, Path]]:
    """
    Finds the phase 2 cells under ``root`` that already have a seed evaluation summary, named
    ``P2_MAPPO_PS_<Env>``, ``P2_MAPPO_noPS_<Env>``, ``P2_MAPPO_PS_Team_<Env>`` and ``P2_CPPO_<Env>``.

    Args:
        root (PathLike): Directory that holds the tune experiments.
        envs (Iterable[str]): Environment names used in the experiment names.

    Returns:
        cells (Dict[str, Dict[str, Path]]): ``{env: {variant: tune_dir}}`` for the finished cells.
    """

    # Keep the cells whose seed evaluation has been aggregated
    cells: Dict[str, Dict[str, Path]] = {}
    for env in envs:
        for variant, prefix in PAPER_EXPERIMENT_PREFIXES.items():
            tune_dir = Path(root) / f"{prefix}_{env}"
            if (tune_dir / SEED_EVAL_SUBDIR / SEED_EVAL_SUMMARY_FILENAME).exists():
                cells.setdefault(env, {})[variant] = tune_dir

    return cells

def gap_decomposition(
    cells: Mapping[str, Mapping[str, PathLike]],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes the three effects per environment as paired differences of network returns over
    the shared seeds, with ``1 - alpha`` intervals and Holm-Bonferroni corrected p-values over all
    effects in the table. An effect is skipped while one of its two cells is missing.

    Args:
        cells (Mapping[str, Mapping[str, PathLike]]): ``{env: {variant: tune_dir}}``, e.g. from
            :func:`paper_cells`.
        alpha (float): Significance level of the intervals and the Holm correction.

    Returns:
        results (Dict[str, Dict[str, Any]]): Per environment, ``returns`` (``{variant: mean
            network return}``), ``locked_configs`` (``{variant: name}``) and ``effects``
            (``{effect: {delta, relative, ci_low, ci_high, p_value, p_holm, reject, n, mean_a,
            mean_b}}``), where ``relative`` is ``delta / |mean_b|``.
    """

    # Load the locked config of every cell
    results: Dict[str, Dict[str, Any]] = {}
    p_values: Dict[str, float] = {}
    for env, variants in cells.items():
        loaded = {variant: load_locked_cell(tune_dir) for variant, tune_dir in variants.items()}
        env_result = {
            "returns": {variant: cell["mean"] for variant, cell in loaded.items()},
            "locked_configs": {variant: cell["locked_config"] for variant, cell in loaded.items()},
            "effects": {},
        }

        # Compute each effect whose two cells are available
        for effect, (variant_a, variant_b) in EFFECTS.items():
            if variant_a not in loaded or variant_b not in loaded:
                continue
            diff = paired_difference(
                loaded[variant_a]["per_seed"], loaded[variant_b]["per_seed"],
                alpha=alpha, label=f"{env}/{effect}",
            )
            env_result["effects"][effect] = {
                "delta": diff.delta,
                "relative": diff.delta / abs(diff.mean_b),
                "ci_low": diff.ci_low,
                "ci_high": diff.ci_high,
                "p_value": diff.p_value,
                "n": diff.n,
                "mean_a": diff.mean_a,
                "mean_b": diff.mean_b,
            }
            p_values[f"{env}/{effect}"] = diff.p_value
        results[env] = env_result

    # Correct the p-values of all effects in the table together
    if p_values:
        for label, holm in holm_bonferroni(p_values, alpha=alpha).items():
            env, effect = label.split("/", 1)
            results[env]["effects"][effect]["p_holm"] = holm.p_adjusted
            results[env]["effects"][effect]["reject"] = holm.reject

    return results
