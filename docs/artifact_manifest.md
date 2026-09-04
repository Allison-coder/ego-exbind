# Artifact Manifest

This repository contains cleaned code and small paper-facing research artefacts. Large datasets, videos, checkpoints, cached features, and trained model weights are excluded.

## Included Artefacts

| Artefact | Location | Notes |
| --- | --- | --- |
| Dataset and checkpoint setup notes | `data/README.md` | Instructions for external EgoClip, EK100-MIR, and EgoVLPv2 resources |
| Experiment configurations | `configs/` | Public YAML files with placeholder paths and configurable hyperparameters |
| Exposure Ledger pipeline | `src/exposure/`, `scripts/01_build_exposure_ledger.sh` | Builds exposure counts and SC/UC/UA splits |
| Retrieval analysis code | `src/retrieval/`, `scripts/02_run_zero_shot_retrieval.sh` | Computes retrieval metrics and exposure analyses |
| Binding probe code | `src/binding/`, `scripts/03_run_binding_probe.sh` | Evaluates noun-swap and verb-swap diagnostics |
| Adapter intervention code | `src/interventions/`, `scripts/04_run_adapter_interventions.sh` | Validates Tables 4.3-4.5 |
| Counterfactual exposure code | `src/interventions/exposure_reweighting.py`, `scripts/05_run_counterfactual_exposure.sh` | Evaluates Table 4.6 |
| Paper-facing intervention tables | `results/tables/` | Released CSV summaries for Tables 4.3-4.5 |

## Excluded Artefacts

The following artefacts are intentionally not tracked:

- Raw EgoClip and EPIC-KITCHENS-100 data
- Video files and extracted feature caches
- EgoVLPv2 checkpoints and trained intervention weights
- Scheduler logs, intermediate outputs, and machine-specific paths
