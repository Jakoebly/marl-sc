# Project Context: Multi-Agent Reinforcement Learning for Supply Chain Inventory Management

## Goal

This research project investigates whether multi-agent reinforcement learning (MARL) can learn effective decentralized replenishment policies for a multi-warehouse, multi-SKU distribution network. Each warehouse is controlled by an independent RL agent that must decide how much of each product to reorder at every timestep. The overarching objective is to minimize total supply chain cost — comprising inventory holding costs, lost-sales penalties, and shipment costs — under stochastic demand and non-trivial fulfillment logistics.

The project compares two MARL paradigms:

- **IPPO (Independent PPO):** Each warehouse agent learns its own policy using only local information, with no explicit coordination mechanism beyond the shared environment dynamics.
- **MAPPO (Multi-Agent PPO):** A centralized-training, decentralized-execution (CTDE) approach where agents still act based on local observations at execution time, but during training each agent's value function (critic) has access to a global view of the entire system state.

Both approaches optionally support parameter sharing, where all warehouse agents share the same neural network weights and are distinguished only by their individual observations.

---

## Supply Chain Model

### Network Topology

The supply chain is a **single-echelon distribution network** consisting of:

- **Warehouses** (typically 3): Peer-level facilities that hold inventory and fulfill customer demand. There is no upstream manufacturer or multi-tier hierarchy modeled explicitly — replenishment orders are placed to an implicit external supplier with no capacity constraints.
- **Demand regions** (typically 3, one per warehouse): Geographic areas where customer orders originate. Each warehouse has a "home region" (its closest region by distance), but any warehouse can ship to any region.
- **SKUs** (typically 3): Distinct product types, each with its own weight, demand patterns, and cost characteristics.

The environment configurations model realistic geographic setups, e.g., two European warehouses and one US warehouse ("2EU_1US"), or three European warehouses ("3EU"), with distances and shipment costs reflecting the geographic separation.

### Agent Decisions

Each warehouse agent makes exactly one type of decision per timestep: **how many units of each SKU to order from its external supplier**. The action is a continuous vector (one value per SKU) that gets rescaled from the neural network's output range to integer order quantities between zero and a configured maximum. There are no production, pricing, or routing decisions — the agents only control replenishment.

### Demand Model

Customer demand is stochastic and generated independently for each region at every timestep via a pluggable demand sampler. Available options:

1. **Poisson (synthetic):** For each region, the number of customer orders per timestep is drawn from a Poisson distribution. Each order includes a random subset of SKUs (each SKU included independently with a configured probability), and the quantity per SKU is drawn from a SKU-specific Poisson distribution. The Poisson rate parameters differ across regions and SKUs, creating heterogeneous demand patterns.

2. **Empirical (real-world):** Historical order data from a CSV is replayed. A random contiguous time window is sampled at the start of each episode, and the corresponding historical orders are fed to the environment timestep by timestep. Note that this mode exists for evaluation purposes only since the real-world data does not contain sufficient timesteps for meaningful learning.

An important modeling nuance: demand arrives as individual customer orders (each requesting potentially multiple SKUs), not as aggregate demand per SKU. This matters because fulfillment and shipment costs are computed at the order level.

### Demand Fulfillment and Transshipment

When customer orders arrive, they must be allocated to warehouses for fulfillment via a pluggable demand allocator. Available options:

1. **Greedy allocator:** Processes each order by ranking warehouses by their total shipment cost to the order's region (cheapest first), then fulfilling as much as possible from the cheapest warehouse, splitting to the next cheapest if needed, and continuing until the order is fully fulfilled or a configurable maximum number of warehouse splits is reached.
2. **LP allocator:** For future use, an LP-based allocator is planned that allocates orders by solving a simple LP. This is not yet implemented.

This means **lateral transshipment** between warehouses occurs implicitly: a warehouse in Europe may ship to a US region if it has inventory and the primary US warehouse is stocked out. Order splitting allows partial fulfillment from multiple warehouses but incurs separate shipment costs for each split.

Any demand that remains unfulfilled after exhausting all available warehouses becomes **lost sales**.

### Lead Times

Replenishment orders placed by agents do not arrive instantly. Each order has a **lead time** during which the ordered units are in transit and unavailable. Agents must anticipate future demand and order ahead to compensate for this delay. Orders that would arrive after the episode ends are discarded. The lead time is determined by a pluggable lead time sampler. Available options:

1. **Uniform:** Lead times are sampled independently per SKU from a discrete uniform distribution over a configurable [min, max] range, introducing stochastic delivery delays.
2. **Custom (deterministic):** Lead times are fixed values specified per (warehouse, SKU) pair, providing a deterministic delay structure (e.g., all lead times set to 3 timesteps).

### Cost Structure

The reward signal is computed by a pluggable reward calculator. The available option is a **cost-based** reward calculator where the reward equals the **negative of total supply chain costs**, consisting of four components (with configurable weights):

1. **Holding cost:** Incurred per unit of inventory held at each warehouse at each timestep, scaled by SKU weight. Penalizes excess inventory.

2. **Penalty cost (lost sales):** Incurred for each unit of demand that could not be fulfilled. Configured per SKU, with higher-value or harder-to-source SKUs carrying higher penalties. Lost sales are attributed back to warehouses via a pluggable lost sales handler. Available options:
   - *Closest:* All lost sales in a region are assigned to the nearest warehouse.
   - *Shipment-proportional:* Lost sales are distributed to warehouses proportionally to how many units they shipped to that region (the more you served a region, the more of its lost sales you bear). Falls back to the closest warehouse if no shipments occurred.
   - *Cost-based (softmax):* Lost sales are distributed based on shipment cost proximity using a softmax over inverse costs, so cheaper-to-reach warehouses bear more responsibility. A temperature parameter controls how concentrated or uniform the assignment is.

3. **Outbound shipment cost:** Incurred when shipping fulfilled orders from warehouses to customer regions. Comprises a fixed per-order component and a variable per-weight component, both configured as matrices over (warehouse, region) pairs, reflecting different geographic distances.

4. **Inbound shipment cost:** Incurred when receiving replenishment orders from the external supplier. Also has fixed and variable components, configured per (warehouse, SKU) pair.

The reward can operate in two scopes:
- **Team (cooperative):** All warehouses receive the same reward, equal to the negative sum of all warehouses' costs. This encourages globally optimal behavior.
- **Agent (individual):** Each warehouse receives only its own negative cost as reward.

### Episode Structure

Each episode runs for a fixed number of timesteps (typically 100). Within each timestep, the sequence of events is:

1. Agents place replenishment orders (actions are applied).
2. Previously ordered inventory that has completed its lead time arrives and is added to stock.
3. New customer demand is sampled.
4. Demand is allocated across warehouses and fulfilled from inventory.
5. Inventory levels are updated.
6. Observation features (demand signals, shipment info, stockouts, forecasts) are computed.
7. Lost sales are assigned to warehouses.
8. Costs are calculated and rewards are returned.

### Agent Observations

Each warehouse observes the following local features (all per-SKU, i.e. one value per SKU):

1. **Current inventory levels** — on-hand stock for each SKU.
2. **Pending orders** — total units currently in transit (ordered but not yet arrived).
3. **Incoming demand (home region)** — total demand originating from the warehouse's home region.
4. **Units shipped to home region** — how much this warehouse actually shipped to its home region.
5. **Units shipped to other regions (away)** — how much this warehouse shipped to non-home regions (transshipment).
6. **Stockout** — unmet demand in the home region (demand minus what was shipped).
7. **Rolling demand mean** — 5-timestep rolling average of home-region demand.
8. **Demand forecast** — exponential moving average of home-region demand (smoothing factor 0.3).

Additionally, **aggregate features** are available for most feature groups (inventory, pending orders, demand, shipped away, rolling mean, demand forecast). These are scalar summaries across all SKUs — such as log-scaled totals or ratios — appended alongside the per-SKU values in each feature group. They provide the agent with a sense of overall scale that per-SKU breakdowns alone may not convey clearly, especially under ratio normalization where per-SKU values are expressed as fractions.

The **global observation** is the concatenation of all warehouses' local observations, providing full system visibility. In IPPO, the critic only sees local observations. In MAPPO, the critic sees the global observation (CTDE paradigm) while the actor still only sees local observations.

Observation normalization options include ratio normalization (per-SKU features expressed as fractions of their totals, with aggregate features log-scaled) and mean-std normalization.

### Pluggable Components Summary

The environment is modular — key dynamics are governed by interchangeable components, each with multiple options:

| Component | Available Options |
|---|---|
| **Demand sampler** | Poisson (synthetic), Empirical (real-world data) |
| **Demand allocator** | Greedy (cheapest-first with order splitting) |
| **Lead time sampler** | Uniform (stochastic per-SKU), Custom (deterministic per warehouse-SKU pair) |
| **Lost sales handler** | Closest warehouse, Shipment-proportional, Cost-based softmax |
| **Reward calculator** | Cost-based (with configurable cost weights, team or agent scope) |

This modularity allows systematic ablation studies to understand how each modeling choice affects learned policies.

---

## Key Challenges and Research Questions

- **Credit assignment:** With a shared team reward, how can individual agents learn which of their actions contributed to good or bad outcomes, especially when costs arise from complex interactions (e.g., one warehouse's stockout causes transshipment costs at another)?
- **Coordination without communication:** Warehouses have no explicit communication channel. In IPPO they must implicitly coordinate through the shared environment. MAPPO's centralized critic provides indirect coordination during training — is this sufficient?
- **Demand uncertainty and lead times:** Agents must manage the classic newsvendor-type tradeoff between overstocking (holding costs) and understocking (penalty costs), compounded by multi-day lead times that delay inventory replenishment.
- **Transshipment dynamics:** The greedy allocator creates coupling between agents — one warehouse's inventory decision affects which orders get routed to other warehouses. Agents must learn to account for these spillover effects.
- **Heterogeneous demand across regions:** Demand patterns vary significantly across regions (e.g., one region may have very low demand), requiring agents to learn region-specific stocking strategies.
- **Scalability:** Does the approach scale gracefully as the number of warehouses, SKUs, or regions increases?
