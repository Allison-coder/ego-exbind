# Ego-ExBind: Diagnosing Pretraining Exposure and Verb–Noun Binding in Egocentric Video–Text Retrieval

**Ego-ExBind** is a diagnostic protocol for studying the relationship between pretraining exposure and verb–noun binding in egocentric video–text retrieval.

> **TL;DR:** Ego-ExBind separates pretraining exposure from compositional binding and shows that stronger exposure-associated retrieval performance does not necessarily imply stronger verb–noun binding.

## 📁 Repository Structure

```text
ego-exbind/
├── configs/        # Experiment configurations
├── data/           # Dataset and checkpoint setup
├── docs/           # Experiment and artefact documentation
├── results/        # Released result artefacts
├── scripts/        # Experiment entry points
└── src/
    ├── exposure/       # Exposure Ledger construction
    ├── retrieval/      # Retrieval and exposure analyses
    ├── binding/        # Verb–noun binding diagnostics
    └── interventions/  # Representation- and score-level interventions
```

## 🛠️ Environment Preparation

The code was developed and tested with Python 3.8.13 and PyTorch 1.13.1+cu116.

```bash
git clone https://github.com/Allison-coder/ego-exbind.git
cd ego-exbind

conda create --name egoexbind python=3.8.13 -y
conda activate egoexbind
pip install -r requirements.txt
```

Experiment configurations are provided in [`configs/`](configs/).

## 📦 Data and Pretrained Model

The experiments use:

- **EgoClip** for measuring pretraining exposure;
- **EPIC-KITCHENS-100 Multi-Instance Retrieval (EK100-MIR)** for downstream evaluation;
- **EgoVLPv2** as the pretrained video–language backbone.

Large datasets, videos, checkpoints, cached features, and trained model weights are not redistributed in this repository.

See [`data/README.md`](data/README.md) for dataset and checkpoint setup.

## 🔬 Experiments

The main experiment pipeline consists of:

1. **Exposure Ledger construction**  
   Build verb, noun, and verb–noun composition exposure statistics from EgoClip.
2. **Zero-shot retrieval and exposure analysis**  
   Evaluate frozen EgoVLPv2 on EK100-MIR and analyse SC/UC/UA, pair exposure, PMI, and exposure-controlled regressions.
3. **Verb–noun binding diagnostics**  
   Evaluate controlled noun-swap and verb-swap probes.
4. **Diagnostic interventions**  
   Evaluate domain adaptation, verb-focused hard negatives, noun-margin protection, and exposure decorrelation.
5. **Counterfactual exposure reweighting**  
   Compare original, zeroed, and shuffled exposure signals while keeping video and text representations frozen.

The corresponding entry points are:

```text
scripts/
├── 01_build_exposure_ledger.sh
├── 02_run_zero_shot_retrieval.sh
├── 03_run_binding_probe.sh
├── 04_run_adapter_interventions.sh
└── 05_run_counterfactual_exposure.sh
```

Detailed arguments and the mapping between dissertation experiments, scripts, and outputs are documented in [`docs/experiment_mapping.md`](docs/experiment_mapping.md).

## 📊 Results

Machine-readable result tables and released dissertation figures are stored under:

```text
results/
├── figures/
└── tables/
```

The CSV files contain the numerical results underlying the corresponding dissertation tables; formatting and rounding may differ from the typeset LaTeX tables.

Released and excluded research artefacts are documented in [`docs/artifact_manifest.md`](docs/artifact_manifest.md).

The Exposure Ledger freeze protocol is documented in [`docs/exposure_freeze.md`](docs/exposure_freeze.md).

## 🙏 Acknowledgements

This work builds upon **EgoVLPv2**, **EgoClip**, and **EPIC-KITCHENS-100**. We thank the authors and maintainers of these projects and datasets.

## 🎓 Citation

Citation information will be added upon paper release.
