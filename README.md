# SIRA -- Learning the SIR Model from Stochastic Simulations

GSoC 2026 evaluation tasks for [HumanAI Foundation](https://humanai.foundation/).

## Project

Three-stage pipeline to recover the deterministic SIR equations from noisy stochastic epidemic data:
1. **Simulate**: Gillespie SSA generates 120,000+ stochastic trajectories
2. **Learn**: Neural ODE learns mean dynamics conditioned on (beta, gamma)
3. **Recover**: SINDy and PySR extract symbolic ODE from the learned dynamics

## Repository Structure

```
src/
  simulation/
    gillespie.py      # Gillespie SSA, ensemble simulation, deterministic validation
    dataset.py        # Full dataset generation, train/val/test splits
  models/
    neural_ode.py     # Parameter-conditioned Neural ODE with conservation penalty
  symbolic/
    recovery.py       # Derivative extraction, SINDy, PySR pipelines

notebooks/
  01_gillespie_simulation.ipynb    # Task 1: Stochastic simulation + validation
  02_neural_ode.ipynb              # Task 2: Neural ODE training + evaluation
  03_symbolic_recovery.ipynb       # Task 3: Symbolic recovery (SINDy + PySR)

data/                              # generated (not committed)
  sir_dataset.npz                  # full dataset

proposal/
  proposal.txt                     # GSoC proposal (LaTeX)
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

Run notebooks in order:

1. `01_gillespie_simulation.ipynb` -- generates 120,000 stochastic trajectories, validates against ODE, produces dataset
2. `02_neural_ode.ipynb` -- trains Neural ODE on mean trajectories, evaluates on test set
3. `03_symbolic_recovery.ipynb` -- applies SINDy and PySR to recover symbolic equations

## Model

| Component | Specification |
|---|---|
| Simulator | Gillespie SSA (exact stochastic simulation) |
| Neural ODE | MLP: 3 layers x 64 units, SiLU, input [s,i,r,beta,gamma] |
| ODE Solver | torchdiffeq, Dormand-Prince (dopri5) |
| Conservation | Soft penalty on ds/dt + di/dt + dr/dt |
| SINDy | Physics-motivated library, STLS sparse regression |
| PySR | Genetic programming, operators {+,-,*,/} |

## Target Equations

```
ds/dt = -beta * s * i
di/dt =  beta * s * i - gamma * i
dr/dt =  gamma * i
```

## Parameter Grid

| Parameter | Range | Values |
|---|---|---|
| beta | [0.1, 1.0] | 10 |
| gamma | [0.05, 0.5] | 10 |
| N | {500, 1000, 5000, 10000} | 4 |
| I0 | {1, 5, 10} | 3 |
| **Total** | 1,200 combinations x 100 realizations | **120,000 trajectories** |

## References

- Chen et al. (2018) -- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
- Brunton et al. (2016) -- [Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems](https://www.pnas.org/doi/10.1073/pnas.1517384113)
- Cranmer (2023) -- [PySR: Interpretable ML for Science](https://arxiv.org/abs/2305.01582)
- Vasilyeva et al. (2026) -- [Neural ODEs, KAN-ODEs and SINDy for Epidemic Processes](https://arxiv.org/abs/2601.09811)
- Pani et al. (2025) -- A Novel Scientific Machine Learning Method for Epidemiological Modelling. *eScience*, 2025.
