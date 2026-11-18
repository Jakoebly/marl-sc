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
