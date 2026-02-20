# Macro Intervention Effects

Code for reproducing the results in the paper on "The Impact of Micro-level User Interventions on Macro-level Misinformation Spread". This repository implements a **Continuous-Time Independent Cascade (CTIC)** model for misinformation spread on social networks and evaluates three intervention strategies: **Prebunking**, **Contextualization**, and **Nudging**, both via simulation and quenched mean-field theory.

## Directory Structure

```
.
├── CTIC.py                              # CTIC model (simulation engine)
├── load_graph.py                        # Graph loading (Nikolov / randomized)
├── estimate_eta_lam.py                  # Estimation of η, λ (fitting to Twitter cascades)
├── estimate_epsilon.py                  # Estimation of intervention effect ε (from intervention datasets)
├── quenched_mean_field.py               # Critical threshold via quenched mean-field theory
├── intervention_analysis.py             # Heatmaps with fixed η (CLI)
├── intervention_analysis_varying_eta.py # Heatmaps with varying η (CLI)
├── single_vs_combined.py                # Single vs. combined intervention comparison
├── preprocess_intervention_dataset.py   # Preprocessing of intervention datasets
├── plot_results.py                      # Visualization of results (Jupyter notebook style)
├── util.py                              # Shared utility functions
├── requirements.txt
├── data/
│   ├── nikolov/                         # Nikolov social network data
│   ├── twitter_diffusion_dataset/       # Twitter cascade data (for parameter estimation)
│   ├── intervention_dataset/            # Intervention experiment datasets (for ε estimation)
└── results/
    ├── nikolov/                         # Results on Nikolov graph
    └── test/                            # Results on test graph
```

## Setup

```bash
pip install -r requirements.txt
```

## File Descriptions

### Core Model

| File | Description |
|------|-------------|
| `CTIC.py` | CTIC model implementation. Event-driven simulation of misinformation spread with support for Prebunking (pre-exposure), Contextualization (post-exposure), and Nudging (population-wide) interventions. |
| `load_graph.py` | Builds and caches the Nikolov graph. `Nikolov_susceptibility_graph()` returns the original graph; `randomized_nikolov_graph()` returns a graph with shuffled susceptibility values. |

### Parameter Estimation

| File | Description |
|------|-------------|
| `estimate_eta_lam.py` | Estimates misinformation contagiousness η and delay rate λ by fitting simulated cascade curves to Twitter cascade data. |
| `estimate_epsilon.py` | Estimates intervention effect ε from intervention experiment data (Pennycook 2020/2021, Fazio 2020, Basol 2021, Drolsbach 2024). |
| `quenched_mean_field.py` | Computes critical transmission threshold from the adjacency matrix and susceptibility vector using quenched mean-field theory. |

### Intervention Simulation

| File | Description |
|------|-------------|
| `intervention_analysis.py` | Generates heatmaps over intervention parameter space with fixed η. Explores (ε_pre, δ_pre) for Prebunking and (ε_ctx, φ_ctx) for Contextualization. |
| `intervention_analysis_varying_eta.py` | Generates heatmaps over (ε, η) space. Supports Prebunking, Contextualization, and Nudging. |
| `single_vs_combined.py` | Compares five conditions: no intervention, Prebunking, Contextualization, Nudging, and combined intervention; saves prevalence distributions. |

### Preprocessing and Visualization

| File | Description |
|------|-------------|
| `preprocess_intervention_dataset.py` | Converts multiple intervention datasets (Pennycook, Fazio, Basol, Drolsbach) into a unified format. |
| `plot_results.py` | Visualization script (Jupyter notebook style) for heatmaps, violin plots, line plots, and critical curves. |
| `util.py` | Shared utilities: seed user selection, heatmap plotting, violin plots, relative suppression computation. |

## Usage

### Intervention Simulation

```bash
# Prebunking (ε_pre × δ_pre heatmap)
python intervention_analysis.py -i prebunking -g nikolov -s 10 --save_data --target_selection random

# Contextualization (ε_ctx × φ_ctx heatmap)
python intervention_analysis.py -i contextualization -g nikolov -s 10 --save_data

# Prebunking (ε_pre × η heatmap)
python intervention_analysis_varying_eta.py -i prebunking -g nikolov -s 10 --save_data

# Contextualization (ε_ctx × η heatmap)
python intervention_analysis_varying_eta.py -i contextualization -g nikolov -s 10 --save_data

# Nudging (ε_nud × η heatmap)
python intervention_analysis_varying_eta.py -i nudging -g nikolov -s 10 --save_data
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `-i`, `--intervention` | Intervention type (`prebunking`, `contextualization`) | required |
| `-g`, `--graph` | Graph name (`test`, `nikolov`, `randomized_nikolov`) | `test` |
| `-s`, `--simulations` | Number of simulations per grid point | `10` |
| `--seed` | Random seed | `42` |
| `--save_data` | Save results as `.npy` files | `false` |
| `--target_selection` | Target selection strategy (`random`, `high_degree`, `high_susceptible`, `cocoon`) | `random` |
| `--eta_scale` | Scale factor for η (`0.5`, `1`, `2`) | `1` |


## Datasets

| Directory | Description |
|-----------|-------------|
| `data/nikolov/` | Nikolov social network: friend links (`anonymized-friends.json`) and susceptibility scores (`measures.tab`). A cached graph is written as `nikolov_graph.pkl`. |
| `data/twitter_diffusion_dataset/` | Twitter information diffusion cascades, used for estimating η and λ. |
| `data/intervention_dataset/` | Intervention experiment datasets (Pennycook 2020/2021, Fazio 2020, Basol 2021, Drolsbach 2024), used for estimating ε. |

## Notation

| Symbol | Variable | Meaning |
|--------|----------|---------|
| $\eta$ | `eta` | Misinformation contagiousness |
| $\lambda$ | `lam` | Exponential delay parameter |
| $\varepsilon_pre$ | `epsilon_pre` | Prebunking effect strength |
| $\varepsilon_ctx$ | `epsilon_ctx` | Contextualization effect strength |
| $\varepsilon_nud$ | `epsilon_nud` | Nudging effect strength |
| $\delta_pre$ | `delta_pre` | Fraction of nodes targeted by Prebunking |
| $\phi_ctx$ | `intervention_threshold` | Contextualization trigger parameter |
