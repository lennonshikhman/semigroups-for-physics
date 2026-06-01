# Semigroup Consistency for Learned Physics Simulators

This repository contains the experimental code for the ICML 2026 Workshop on AI for Physics paper:

**Semigroup Consistency as a Diagnostic for Learned Physics Simulators**  
Lennon J. Shikhman  
*ICML 2026 Workshop on AI for Physics*  
OpenReview: https://openreview.net/forum?id=MeAFOZnrvM

The code implements lightweight experiments for evaluating semigroup consistency in learned time-evolution models. The focus is on using semigroup error as a post hoc, model-agnostic diagnostic for learned physics simulators, rather than as a guarantee of correctness or a universally beneficial training objective.

## Overview

Learned PDE simulators are often evaluated with one-step prediction error or long-horizon rollout error. These metrics are useful, but they do not directly test whether a learned model behaves like a coherent autonomous time-evolution map.

This repository studies **semigroup consistency** as an additional diagnostic. For autonomous, state-complete systems, exact solution maps satisfy a structural identity: evolving by a total time `s + t` should agree with evolving first by `s` and then by `t`. The experiments measure how closely learned simulators satisfy this identity on held-out trajectories.

The paper finds that unseen semigroup error is positively associated with rollout degradation. In the reported trajectory-level analysis, the Spearman correlation between unseen semigroup error and rollout degradation is `ρ = 0.635`, with 95% confidence interval `[0.621, 0.649]`.

Semigroup consistency should be interpreted as a structural warning signal. A small semigroup error does **not** prove that a model is physically correct, accurate, stable, or faithful to the underlying PDE. It should be used alongside one-step error, rollout error, conservation or dissipation checks where appropriate, and other physics-specific diagnostics.

## What is semigroup consistency?

For an autonomous dynamical system with state `u`, the exact solution operator `S_t` satisfies

```text
S_{s+t}(u) = S_t(S_s(u)).
```

A learned time-conditioned simulator `Gθ(u, t)` can be tested against the same identity by comparing

```text
Gθ(u, s + t)
```

with the composed prediction

```text
Gθ(Gθ(u, s), t).
```

The diagnostic reports a normalized semigroup error between the direct and composed predictions. In this codebase, semigroup errors are computed for short time-pair combinations seen during evaluation and for held-out, longer time-pair combinations used to test whether the learned time evolution remains compositionally coherent outside the easiest cases.

## Experiments

The experiments use one-dimensional PDE systems with periodic boundary conditions:

- **Heat equation** with Fourier-based exact evolution.
- **Viscous Burgers equation** with a finite-volume style numerical solver.

The main sweep evaluates:

- in-distribution trajectories,
- out-of-distribution initial-condition spectra, and
- a viscosity-shift setting for Burgers.

Each model is trained both as a baseline predictor and with a simple semigroup-regularized variant. The paper reports mixed effects from semigroup regularization, so the recommended interpretation is diagnostic: semigroup consistency can reveal structural inconsistency, but it is not presented as a guaranteed way to improve training.

## Models

The repository includes two compact time-conditioned learned simulators:

- **Time-conditioned residual ConvNet** (`tc_conv`): a 1D residual convolutional network with sinusoidal time embeddings.
- **Compact 1D Fourier Neural Operator** (`fno1d`): a small Fourier Neural Operator with spectral convolution blocks and time conditioning.

Both models take an input state `u` and a time increment `t`, and return a predicted future state.

## Metrics

The experiment pipeline records the following diagnostics:

- **One-step relative L2 error**: prediction error for a single time increment.
- **Rollout AUC error**: average relative L2 error over an autoregressive rollout trajectory.
- **Final rollout error**: relative L2 error at the final rollout time.
- **Seen semigroup error**: normalized discrepancy for short evaluation time pairs.
- **Unseen semigroup error**: normalized discrepancy for held-out time-pair compositions.
- **Spearman correlation**: trajectory-level association between unseen semigroup error and rollout degradation.
- **Bootstrap confidence intervals** for aggregate summaries and correlations.

These metrics are intended to complement each other. In particular, semigroup error tests a structural property of time composition, while rollout and one-step errors test predictive accuracy against reference trajectories.

## Repository Contents

The repository is organized as a lightweight Python research codebase:

- `experiments.py`: main experiment driver for data generation, training, evaluation, aggregation, and figure export.
- `solvers.py`: random Fourier initial conditions and reference solvers for the heat and viscous Burgers equations.
- `models.py`: time-conditioned residual ConvNet and compact 1D FNO model definitions.
- `metrics.py`: relative L2 error, rollout metrics, and Spearman correlation helper.
- `stats_utils.py`: bootstrap confidence intervals, paired comparisons, Wilcoxon p-values, and effect-size utilities.
- `plots.py`: plotting utilities for semigroup-vs-rollout, rollout curves, seen-vs-unseen semigroup error, and regularization ablations.
- `requirements.txt`: Python package dependencies.

## Installation

Clone the repository:

```bash
git clone https://github.com/lennonshikhman/semigroups-for-physics.git
cd semigroups-for-physics
```

Create and activate a Python environment. For example, with `conda`:

```bash
conda create -n semigroups-for-physics python=3.10
conda activate semigroups-for-physics
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

A GPU-enabled PyTorch installation is recommended for larger sweeps. If you need a CUDA-specific PyTorch build, install PyTorch using the command recommended for your system by the official PyTorch installation selector, then install the remaining dependencies from `requirements.txt`.

## Running Experiments

Run the default experiment sweep:

```bash
python experiments.py
```

For a faster smoke test, set the `QUICK` environment variable:

```bash
QUICK=1 python experiments.py
```

The current script is intentionally compact and does not expose a large command-line interface. It uses constants in `experiments.py` for the number of seeds, epochs, grid size, train/validation/test counts, bootstrap resamples, and whether to run the regularization ablation. Users who modify the experiment configuration should inspect `experiments.py` directly.

## Aggregating/Evaluating Results

Aggregation is performed by the same experiment driver. After training and evaluation, `experiments.py` writes CSV summaries, raw trajectory-level measurements, bootstrap statistics, LaTeX table output, and figures under `outputs/`.

Use:

```bash
python experiments.py
```

or, for a quick end-to-end check:

```bash
QUICK=1 python experiments.py
```

There is no separate all-in-one reproduction or analysis command beyond `experiments.py` in the current repository. If additional command-line arguments are added later, check the script interface with `--help`.

## Outputs

The experiment script writes outputs to `outputs/`, including:

- `outputs/summary.csv`: aggregate metrics grouped by system, model, variant, and regime, with bootstrap confidence intervals.
- `outputs/raw_results.csv`: trajectory-level measurements used for correlation and diagnostic plots.
- `outputs/stats.json`: hypothesis-oriented summary statistics, including bootstrap Spearman confidence intervals and paired comparisons.
- `outputs/paper_tables.tex`: compact LaTeX table export.
- `outputs/run_config.json`: configuration metadata for the run.
- `outputs/README_results.txt`: short pointer to generated results.
- `outputs/fig_sg_vs_rollout.pdf` and `outputs/fig_sg_vs_rollout.png`: unseen semigroup error versus rollout error.
- `outputs/fig_rollout_curves.pdf` and `outputs/fig_rollout_curves.png`: rollout error curves.
- `outputs/fig_seen_vs_unseen.pdf` and `outputs/fig_seen_vs_unseen.png`: seen versus unseen semigroup error.
- `outputs/fig_lambda_ablation.pdf` and `outputs/fig_lambda_ablation.png`: semigroup-regularization ablation plot.

## Reproducing Paper Results

To reproduce the main lightweight experiment pipeline, install the dependencies and run:

```bash
python experiments.py
```

This executes the heat and Burgers experiments across the implemented models, regimes, seeds, and baseline/semigroup-regularized variants, then writes the resulting tables and figures to `outputs/`.

For a quick functional check before launching the full sweep, run:

```bash
QUICK=1 python experiments.py
```

The full paper result `ρ = 0.635` with 95% CI `[0.621, 0.649]` refers to the reported trajectory-level analysis in the paper. Because the repository is a lightweight research implementation with configurable constants, exact numerical reproduction can depend on hardware, PyTorch version, random seeds, and any local changes to experiment settings.

## Citation

If you use this repository or build on the associated paper, please cite:

```bibtex
@inproceedings{
shikhman2026semigroup,
title={Semigroup Consistency as a Diagnostic for Learned Physics Simulators},
author={Lennon J. Shikhman},
booktitle={ICML 2026 Workshop on AI for Physics},
year={2026},
url={https://openreview.net/forum?id=MeAFOZnrvM}
}
```

## Contact

For questions about the paper or repository, please contact Lennon J. Shikhman or open an issue on GitHub.
