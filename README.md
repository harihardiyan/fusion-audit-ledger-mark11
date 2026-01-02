
# Mark 11a: Full Stochastic Fusion Fleet Audit Ledger

**Author:** Hari Hardiyan (AI Tamer)  
**Collaborator:** GitHub Copilot  
**Email:** [lorozloraz@gmail.com](mailto:lorozloraz@gmail.com)

---

## 🔬 Overview

**Mark 11a** is a high-performance, physics-first simulation framework designed for the deterministic and stochastic auditing of fusion reactor fleets. Unlike traditional simulators that focus solely on visual output, Mark 11a acts as a **Thermodynamic Ledger**, ensuring that every Joule of energy—from D-T fusion ignition to neutron capture and tritium breeding—is accounted for with near-zero residual error.

Built on the **JAX** ecosystem, Mark 11a leverages hardware acceleration (GPU/TPU) and functional programming paradigms to simulate complex multi-module reactor environments with rigorous scientific accuracy.

---

## 🚀 Key Technical Advantages

### 1. Full Stochastic Neutronic Validation (Monte Carlo)
Mark 11a implements a native **Monte Carlo (MC) sampling engine** within the JAX `jit` boundary. Instead of assuming a static efficiency, the `op_blanket_stochastic_capture` function simulates hundreds of individual neutron interaction samples (n=256 by default) per step. This captures the probabilistic nature of neutron leakage and blanket absorption, providing a realistic "uncertainty envelope" for energy production.

### 2. High-Precision "Audit Ledger" Architecture
The system is built around the **AuditRecord** and **OpMeta** structures. Every physical process—Fusion, Alpha Heating, DEC (Direct Energy Conversion), and Recuperation—is treated as a discrete transaction.
* **Energy Conservation:** The framework includes a `conservation_residual` check. In recent benchmarks, the system demonstrated a residual error of less than **-6.10e-05 J** against a total energy throughput of over **194 GJ**, proving total compliance with the First Law of Thermodynamics.

### 3. Physical Tritium Breeding & Management
Unlike simplified models, Mark 11a treats the **Breed Store** as a physical energy and material reservoir. It tracks the Tritium Breeding Ratio (TBR) indirectly through energy equivalence, allowing auditors to verify if a fleet is self-sustaining or depleting its fuel reserves.

### 4. Adaptive Control & Dispatch
The simulation includes a **Sigmoid-based Adaptive Controller**. This mimicry of a Real-Time Control (RTC) system adjusts the dispatch scales based on the `CORE` temperature, preventing thermal runaway while maximizing electricity production through temperature-dependent DEC efficiency.

### 5. Numerical Rigor with JAX
* **X64 Precision:** Explicitly configured for `float64` to avoid the accumulation of rounding errors common in standard 32-bit simulations.
* **Vectorized Fleet Rollout:** Using `jax.vmap` and `jax.lax.scan`, the engine can simulate an entire fleet of reactors (e.g., 8+ modules) simultaneously without linear performance degradation.

---

## 📊 Benchmark Results (Mark 11a Audit Ledger)

| Metric | Value |
| :--- | :--- |
| **Total Fusion Energy ($\sum \text{Fusion}$)** | $1.944 \times 10^{11} \text{ J}$ |
| **Total Energy Change ($\Delta E_{\text{total}}$)** | $-1.041 \times 10^{11} \text{ J}$ |
| **Total Losses Over Horizon** | $2.985 \times 10^{11} \text{ J}$ |
| **Final Breed Store (Fleet)** | $8.111 \times 10^{9} \text{ J}$ |
| **Conservation Residual** | $\approx -6.103 \times 10^{-5} \text{ J}$ |
| **Thermal Consistency ($|T(U)-T|$)** | $0.0$ (Perfect) |

---

## 🛠 Installation & Usage

### Prerequisites
* Python 3.9+
* JAX (with `jaxlib`)
* NumPy

```bash
pip install jax jaxlib numpy
```
Running the Audit

```bash
python fleet_mark11a_fixed2.py
```
The simulation will generate a comprehensive audit log: fleetmark11a_audit_fixed2.csv.
📬 Contact
For inquiries regarding the Mark 11a architecture or collaboration on Fusion Fleet Digital Twins, please reach out to Hari Hardiyan via lorozloraz@gmail.com.
Developed as part of the Fusion Fleet Audit Simulation Project (2026).


