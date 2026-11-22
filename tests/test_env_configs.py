"""
Checks every environment config under ``config_files/environments``: it loads and validates, and
a lost unit of every SKU costs more than buying the unit and shipping it to the home region, so
every SKU is worth stocking.
"""

from pathlib import Path

import numpy as np
import pytest

from src.config.loader import load_environment_config

ENV_DIR = Path(__file__).resolve().parents[1] / "config_files" / "environments"
ENV_CONFIGS = sorted(ENV_DIR.rglob("*.yaml"))


def config_id(path: Path) -> str:
    """Short test id: the config path relative to the environments folder."""
    return path.relative_to(ENV_DIR).as_posix()


def test_configs_are_found():
    """Guards against an empty parametrization if the folder moves."""
    assert len(ENV_CONFIGS) > 0

@pytest.mark.parametrize("path", ENV_CONFIGS, ids=config_id)
def test_config_loads(path):
    """The config passes schema validation."""
    load_environment_config(str(path))

@pytest.mark.parametrize("path", ENV_CONFIGS, ids=config_id)
def test_lost_unit_costs_more_than_buying_and_shipping_it(path):
    """
    For every warehouse and SKU: penalty > (inbound rate + outbound rate to the home region) * weight.
    A list penalty is per unit; a scalar penalty is per unit of weight (see CostRewardCalculator).
    """

    env_config = load_environment_config(str(path))
    cost_structure = env_config.cost_structure
    if cost_structure.distances is None:
        pytest.skip("Costs are generated from data, not set in the config")

    weights = np.array(cost_structure.sku_weights, dtype=float) # Shape: (n_skus,)
    penalty = cost_structure.penalty_cost
    penalty_per_unit = np.array(penalty, dtype=float) if isinstance(penalty, list) else penalty * weights

    home_regions = np.argmin(np.array(cost_structure.distances, dtype=float), axis=1) # Shape: (n_warehouses,)
    inbound_rate = np.array(cost_structure.shipment_cost.inbound_variable, dtype=float) # Shape: (n_warehouses, n_skus)
    outbound_rate = np.array(cost_structure.shipment_cost.outbound_variable, dtype=float) # Shape: (n_warehouses, n_regions)
    home_outbound_rate = outbound_rate[np.arange(len(home_regions)), home_regions] # Shape: (n_warehouses,)
    unit_cost = (inbound_rate + home_outbound_rate[:, None]) * weights[None, :] # Shape: (n_warehouses, n_skus)

    assert np.all(penalty_per_unit[None, :] > unit_cost), (
        f"Lost unit is not more expensive than buying and shipping it: penalty {penalty_per_unit}, "
        f"purchase + home shipping {np.round(unit_cost, 3)}"
    )

@pytest.mark.parametrize("path", ENV_CONFIGS, ids=config_id)
def test_stock_outs_are_charged_to_the_home_warehouse(path):
    """
    Every config uses the closest lost-sales handler. Under the shipment handler a warehouse that
    ships nothing to a region pays none of its stock-outs, so with the agent reward the cheapest
    choice for a warehouse is to stock nothing and leave the demand to the others.
    """

    env_config = load_environment_config(str(path))
    assert env_config.components.lost_sales_handler.type == "closest"

@pytest.mark.parametrize("name", ["symmetric", "demand_hetero", "cost_hetero", "sku_hetero", "combined_hetero"])
def test_team_files_differ_from_their_main_file_only_in_the_reward_scope(name):
    """
    The team-reward files of the phase 2 comparison are copies of the five main environments with
    the team reward. Any other difference would mix a second change into the comparison.
    """

    main = load_environment_config(str(ENV_DIR / f"env_{name}_3WH5SKU.yaml")).model_dump()
    team = load_environment_config(str(ENV_DIR / "E3.4_team" / f"env_{name}_team_mappo_ps.yaml")).model_dump()

    assert main["components"]["reward_calculator"]["params"]["scope"] == "agent"
    assert team["components"]["reward_calculator"]["params"]["scope"] == "team"
    team["components"]["reward_calculator"]["params"]["scope"] = "agent"
    assert team == main
