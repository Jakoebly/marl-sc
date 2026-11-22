"""
Tests for the phase 2 gap decomposition (``src/analysis/utils/gap_decomposition.py``): locked
config per cell, conversion of summed returns to network returns, and the three paired effects.
"""

from pathlib import Path

import pytest
import yaml

from src.analysis.utils.gap_decomposition import (
    gap_decomposition,
    load_locked_cell,
    network_return_factor,
    paper_cells,
)
from src.experiments.utils.experiment_utils import SEED_EVAL_SUBDIR, SEED_EVAL_SUMMARY_FILENAME

SEEDS = (100, 200, 300, 400, 500)

# Network returns per seed of the locked config of each variant
SHARED_OWN = (-420.0, -418.0, -422.0, -421.0, -419.0)    # mean -420
SEPARATE_OWN = (-425.0, -424.0, -426.0, -427.0, -423.0)  # mean -425
SHARED_TEAM = (-410.0, -409.0, -411.0, -412.0, -408.0)   # mean -410
CENTRAL = (-405.0, -403.0, -406.0, -408.0, -403.0)       # mean -405


def write_env_config(path: Path, scope: str, team_weight=None, n_warehouses: int = 3) -> Path:
    """Saved env config with only the fields the decomposition reads."""

    params = {"scope": scope, "team_weight": team_weight}
    config = {"environment": {"n_warehouses": n_warehouses, "components": {"reward_calculator": {"params": params}}}}
    path.write_text(yaml.safe_dump(config))
    return path

def make_cell(root: Path, name: str, configs, scope: str = "agent") -> Path:
    """
    Seed evaluation of one cell: ``configs`` is a list of (config name, summed returns per seed).
    Every config gets one seed run with a saved env config.
    """

    seed_dir = root / name / SEED_EVAL_SUBDIR
    summary = {"configs": []}
    for config_name, returns in configs:
        summary["configs"].append({
            "name": config_name,
            "mean": sum(returns) / len(returns),
            "per_seed": [{"seed": seed, "value": value} for seed, value in zip(SEEDS, returns)],
        })
        run_dir = seed_dir / config_name / f"{config_name}_Seed{SEEDS[0]}"
        run_dir.mkdir(parents=True)
        write_env_config(run_dir / "env_config.yaml", scope)
    (seed_dir / SEED_EVAL_SUMMARY_FILENAME).write_text(yaml.safe_dump(summary))
    return root / name

def make_symmetric_cells(root: Path, variants=("shared_own", "separate_own", "shared_team", "central")):
    """The four Symmetric cells; the team cell stores summed returns, i.e. 3 x the network return."""

    if "shared_own" in variants:
        worse = tuple(value - 30.0 for value in SHARED_OWN)
        make_cell(root, "P2_MAPPO_PS_Symmetric", [("01_trainable_a", SHARED_OWN), ("02_trainable_b", worse)])
    if "separate_own" in variants:
        make_cell(root, "P2_MAPPO_noPS_Symmetric", [("01_trainable_c", SEPARATE_OWN)])
    if "shared_team" in variants:
        summed = tuple(3.0 * value for value in SHARED_TEAM)
        make_cell(root, "P2_MAPPO_PS_Team_Symmetric", [("01_trainable_d", summed)], scope="team")
    if "central" in variants:
        make_cell(root, "P2_CPPO_Symmetric", [("01_trainable_e", CENTRAL)])


def test_network_return_factor(tmp_path):
    """A summed return counts the network return once (agent), W times (team) or in between (mixed)."""

    assert network_return_factor(write_env_config(tmp_path / "a.yaml", "agent")) == 1.0
    assert network_return_factor(write_env_config(tmp_path / "t.yaml", "team")) == 3.0
    assert network_return_factor(write_env_config(tmp_path / "m.yaml", "mixed", team_weight=0.5)) == 2.0

def test_locked_cell_is_the_best_config_in_network_returns(tmp_path):
    """The locked config is the best mean; team returns are divided by the number of warehouses."""

    make_symmetric_cells(tmp_path)
    shared = load_locked_cell(tmp_path / "P2_MAPPO_PS_Symmetric")
    team = load_locked_cell(tmp_path / "P2_MAPPO_PS_Team_Symmetric")

    assert shared["locked_config"] == "01_trainable_a" and shared["mean"] == pytest.approx(-420.0)
    assert team["factor"] == 3.0 and team["mean"] == pytest.approx(-410.0)
    assert team["per_seed"] == pytest.approx(dict(zip(SEEDS, SHARED_TEAM)))

def test_three_effects_are_paired_differences_of_network_returns(tmp_path):
    """sharing = separate - shared, own_cost = team - shared, local_view = central - team."""

    make_symmetric_cells(tmp_path)
    cells = paper_cells(tmp_path)
    assert set(cells) == {"Symmetric"} and len(cells["Symmetric"]) == 4

    result = gap_decomposition(cells)["Symmetric"]
    effects = result["effects"]
    assert effects["sharing"]["delta"] == pytest.approx(-5.0)
    assert effects["own_cost"]["delta"] == pytest.approx(10.0)
    assert effects["local_view"]["delta"] == pytest.approx(5.0)
    assert effects["own_cost"]["relative"] == pytest.approx(10.0 / 420.0)
    for effect in effects.values():
        assert effect["ci_low"] < effect["delta"] < effect["ci_high"]
        assert effect["p_holm"] >= effect["p_value"] and effect["n"] == len(SEEDS)
    assert result["returns"]["central"] == pytest.approx(-405.0)

def test_effects_with_a_missing_cell_are_skipped(tmp_path):
    """While only the shared cells are done, only the own-cost effect is computed."""

    make_symmetric_cells(tmp_path, variants=("shared_own", "shared_team"))
    result = gap_decomposition(paper_cells(tmp_path))["Symmetric"]

    assert set(result["effects"]) == {"own_cost"}
    assert result["effects"]["own_cost"]["delta"] == pytest.approx(10.0)
