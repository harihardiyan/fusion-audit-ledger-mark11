# Mark 11a Energy Audit Methodology (Ledger Specification)

This document details the technical specifications of the Mark 11a energy auditing system for modular fusion reactor fleets.

## 1. Principle of Energy Conservation (First Law Compliance)
Mark 11a utilizes a **Double-Entry Energy Accounting** approach. Every state change in energy must be recorded through the `AuditRecord` structure. The fundamental audit equation is defined as:

$$\Delta E_{total} = \sum E_{fusion} - \sum E_{losses}$$

Where $\Delta E_{total}$ accounts for changes in:
* **Core Energy Level** (Plasma/thermal heat)
* **Neutron Queue** (Energy in transit/transport delay)
* **Breed Store** (Physical potential energy of generated fuel)

## 2. Atomic Operations (Audit Ops)
To ensure maximum transparency, the audit is decomposed into granular atomic operations:

* **OP_FUSION (0):** Recording gross energy input from fusion reactions.
* **OP_DISPATCH (1):** Managing energy flows between modules (Buffer/ELEC).
* **OP_DEC (2):** Direct Energy Conversion of Core heat into electricity.
* **OP_BLANKET (5):** Stochastic neutron capture using Monte Carlo simulations.
* **OP_RECUP (6):** Thermal recovery from reactor walls (blanket/struct) back into the system.
* **OP_PROJECT (10):** Conservation projection to mitigate numerical floating-point drift.

## 3. Stochastic Validation (Monte Carlo Neutronics)
Unlike static models, Mark 11a acknowledges that neutron capture is a probabilistic phenomenon.
* **Methodology:** It utilizes `jax.random` sub-routines to generate neutron capture distributions at every time-step.
* **Audit Trail:** Every neutron leakage event (`J_leak`) is logged as an explicit loss in the ledger.

## 4. Thermal Consistency
The system validates that temperature ($T$) remains a linear function of energy ($U$) relative to heat capacity ($C$) at all times:
$$T = \frac{U}{C}$$
The Mark 11a audit performs a cross-check at the end of every cycle to ensure synchronization between physical energy levels and the system's thermal representation.

## 5. Audit Compliance Thresholds
An audit is considered **PASSED** if:
1. `conservation_residual` < $10^{-3}$ Joules.
2. `thermal_consistency` = 0.
3. `violfloor` (energy floor violation) = False.
4. 
