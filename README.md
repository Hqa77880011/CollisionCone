# CollisionCone

Implementation of **Observability-aware collision-cone perception for small maritime obstacles**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data

Encounters provide timestamps, measured bearing/log-scale, ego position/yaw, sensor noise, initial filter state and prior uncertainty. State order is range, range-rate, bearing, bearing-rate and log object size. Ground-truth state is evaluator-only.

The bundled generator writes synthetic finite-model scenarios as JSONL. Fix the seed and keep the generated input for paired comparisons.

```bash
python -m collisioncone prepare --count 200 --seed 17 --output data/generated.jsonl
```

## Execution and evaluation

```bash
python -m collisioncone run --data data/generated.jsonl --output runs/main
```

## Comparisons and ablations

```bash
python -m collisioncone suite --data data/generated.jsonl --output runs/comparison
```

Use `--methods` to select: `collisioncone`, `always_metric`, `no_ego_compensation`, `cone_only`, `no_nuisance_marginalization`.

## Outputs

`runs/main/collisioncone/records.jsonl` stores per-case results; `metrics.json` stores aggregate measurements. Comparison outputs use one directory per method.

```bash
python -m collisioncone plot --inputs runs/comparison/collisioncone --output figures
```

Plotting writes `figures/summary.csv` and metric PNGs. Statistical intervals can be computed from a retained record file:

```bash
python -m collisioncone.statistics --records runs/main/collisioncone/records.jsonl --output results/intervals.json
```
