# CollisionCone

Implementation of **Observability-aware collision-cone perception for small maritime obstacles**.

Fits stabilized bearing/scale histories, computes measurement-only metric information after nuisance marginalization and branches between a scale-free warning and metric TTC/CPA intervals.

## Install

Use Python 3.11 or newer.

```bash
git clone https://github.com/Hqa77880011/CollisionCone.git
cd CollisionCone
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install.

```bash
python -m pip install -e .
```

## Data preparation

Encounters provide timestamps, measured bearing/log-scale, ego position/yaw, sensor noise, initial filter state and prior uncertainty. State order is range, range-rate, bearing, bearing-rate and log object size. Ground-truth state is evaluator-only.

```bash
python -m collisioncone prepare --count 200 --seed 17 --output data/workload.jsonl
```

The generator creates finite workloads from the formal model. Each JSONL row is one scenario. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m collisioncone run --data data/workload.jsonl --output runs/main
python -m collisioncone suite --data data/workload.jsonl --output runs/comparison
```

Select methods with `--methods`. collisioncone, always_metric, no_ego_compensation, cone_only and no_nuisance_marginalization.

## Evaluation and visualization

High-risk encounter recall, warning lead, false alarms and exposure duration, low-observability false precision, TTC interval coverage, Brier and NLL. False alarms/hour is total false_alarms divided by total duration_hours.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m collisioncone plot --inputs runs/comparison/collisioncone --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

The physical estimator uses a constant-velocity relative-state model and Gaussian posterior, one of the paper's supported density families. The observability threshold and warning operating point must be fixed on independent voyages. The numerical driver accepts detector/tracker histories; it does not infer metric ground truth from image size. Closed-loop vessel trajectories are evaluated separately from encounter warning semantics.


## Uncertainty and paired comparisons

```bash
python -m collisioncone.statistics --records runs/main/collisioncone/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.

## Fixed reproduction setting

Finite noisy bearing/log-size encounters; ego compensation, Schur observability and 2000 posterior samples; perception/warning endpoints.

The checked-in entry point fixes data seeds, methods, budgets and output paths. Inspect its commands without starting any computation:

```bash
python reproduce.py --plan
```

Remove `--plan` to execute the registered pipeline in a fresh output directory. The entry point calls the existing implementation; it does not insert paper results. Installation uses the package instructions above.
