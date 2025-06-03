# Relational-Contrast Simulator (`rc_sim`)

A Rust-based Monte Carlo simulator for exploring the **Relational Contrast** (RC) approach to emergent geometry and gauge theory.

Spacetime structure and gauge interactions are derived from relational data on a graph, using only link weights and contrast tensors.

---

## ✨ Features

- **Graph structure**
  - Complete undirected graphs
  - Link data includes:
    - weight `w ∈ (0, 1]`
    - contrast tensor (3×3×3 real)
    - U(1) gauge phase `θ`

- **Action terms**
  - Entropy term: \( \sum w_{ij} \ln w_{ij} \)
  - Triangle holonomy term: \( \sum_{\triangle} 3\cos(\theta_{ij} + \theta_{jk} + \theta_{ki}) \)
  - Dougal-invariant combination:
    \[
    I = \frac{S - \ln(\Delta t)\sum w}{\Delta t}
    \]

- **AIB Projector**
  - Removes axial, isotropic and cyclic parts of each 3 × 3 × 3 tensor
  - Retains 20 physical degrees of freedom
  - Fully tested and norm-reducing

- **Metropolis sampling**
  - Weight and phase proposals
  - Auto-tuning of δ_w and δ_θ to keep acceptance ≈ 30%
  - Live console output + CSV logging

- **Data analysis**
  - `mc_observables.csv` written every 1000 steps
  - Jupyter notebook provided for exploration and plotting

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR-USERNAME/relational-contrast-sim.git
cd relational-contrast-sim
cargo test                      # run all unit tests
cargo run --bin mc_demo         # run a 10,000-step simulation
```

---

## 📂 Project Structure

```
src/
  projector.rs         # AIB projector logic
  graph.rs             # Graph structure, action terms, Monte Carlo step
  bin/
    mc_demo.rs         # Main Metropolis loop
    graph_demo.rs      # Projects all link tensors
    demo.rs            # Basic tensor projector demo

tests/
  projector_test.rs    # AIB projector behaviour
  graph_test.rs        # Graph construction
  entropy_test.rs      # Entropy term correctness
  dougal_test.rs       # Dougal-invariance verification
  metropolis_test.rs   # Metropolis acceptance
  triangle_test.rs     # Triangle action behaviour

notebooks/
  mc_exploration.ipynb # Plot `mc_observables.csv` with pandas and matplotlib
```

---

## 🧪 Action Terms

### Entropy (Dougal-invariant)

\[
S = \sum w_{ij} \ln w_{ij} \qquad
I = \frac{S - \ln(\Delta t) \sum w_{ij}}{\Delta t}
\]

### Triangle term (U(1))

\[
S_\triangle = \sum_{\triangle} 3\cos(\theta_{ij} + \theta_{jk} + \theta_{ki})
\]

---

## 📈 Observables

During the simulation, the following are printed every 1000 steps:

- Acceptance %
- Current δ_θ
- ⟨cos θ⟩ (average gauge observable)
- Entropy term
- Triangle term
- Total action

These are also written to `mc_observables.csv`.

---

## 📊 Plotting

To explore the results:

```bash
jupyter notebook
```

Then open `notebooks/mc_exploration.ipynb`.

Example:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mc_observables.csv")
df.plot(x="step", y=["avg_cos_theta", "action"])
plt.grid()
plt.show()
```

---

## 🧭 Roadmap

| Stage               | Description |
|--------------------|-------------|
| ✅ Projector tested | AIB projector with norm-reduction and idempotence |
| ✅ Entropy + triangle terms | Fully implemented and tested |
| ✅ Self-tuning δ | Tuners adapt proposal width to target acceptance |
| ✅ CSV logging + notebook | Live analysis in Jupyter |
| ⬜ U(1) histogram analysis | Phase distributions at equilibrium |
| ⬜ SU(3) holonomies | Replace U(1) with full unitary matrices |
| ⬜ Parallel chains | Batch mode for error bars |

---

## 🤝 Contributing

- Rust edition: 2021
- Follows `rustfmt` style
- Lints clean with `cargo clippy`
- Tested with `cargo test`

Open to issues and pull requests.

---

## 📄 Licence

MIT — see `LICENSE`
