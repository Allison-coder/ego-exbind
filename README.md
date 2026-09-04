# Ego-ExBind: Diagnosing Pretraining Exposure and Verb–Noun Binding in Egocentric Video–Text Retrieval

This repository contains the cleaned code and processed research artefacts for **Ego-ExBind**, a diagnostic protocol for studying the relationship between pretraining exposure and verb–noun binding in egocentric video–text retrieval.

Ego-ExBind builds an **Exposure Ledger** from the EgoClip pretraining corpus, partitions EPIC-KITCHENS-100 Multi-Instance Retrieval (EK100-MIR) examples into **Seen Composition (SC)**, **Unseen Composition (UC)**, and **Unseen Atom (UA)** settings, and evaluates binding-sensitive behaviour using controlled noun-swap and verb-swap probes.

Our main finding is that stronger exposure-associated retrieval performance does not necessarily imply stronger verb–noun binding.

## Environment

The code was developed and tested with Python 3.8.13 and PyTorch 1.13.1+cu116.

```bash
git clone https://github.com/Allison-coder/ego-exbind.git
cd ego-exbind

conda create --name egoexbind python=3.8.13 -y
conda activate egoexbind

pip install -r requirements.txt
```

Experiment configurations are stored in `configs/`.

## Data Setup

The experiments use:

- **EgoClip** for measuring pretraining exposure;
- **EPIC-KITCHENS-100 (EK100)** for multi-instance retrieval evaluation;
- **EgoVLPv2** as the pretrained video–language model.

Large datasets, videos, checkpoints, cached features, and trained model weights are not included in this repository. Dataset and checkpoint setup instructions are provided in [`data/README.md`](data/README.md).

## Repository Structure

```text
ego-exbind/
├── configs/        # Experiment configuration files
├── data/           # Dataset and checkpoint setup instructions
├── docs/           # Artefact and dissertation-result mapping documentation
├── results/        # Released dissertation result artefacts
├── scripts/        # End-to-end experiment entry points
└── src/
    ├── exposure/       # Exposure Ledger construction and validation
    ├── retrieval/      # Zero-shot retrieval and exposure analyses
    ├── binding/        # Controlled verb–noun binding diagnostics
    └── interventions/  # Representation- and score-level interventions
```

## Reproducing Experiments

The scripts below are the public entry points for the dissertation experiments. Replace placeholder paths in the YAML files or command arguments with local dataset, checkpoint, and cache locations.

### 1. Exposure Ledger

EgoClip narrations are parsed into verbs, nouns, and verb–noun pairs, then mapped into the EK100 action taxonomy. The resulting Exposure Ledger is frozen and reused across retrieval, binding, and intervention experiments.

```bash
bash scripts/01_build_exposure_ledger.sh configs/exposure.yaml
```

### 2. Zero-Shot Retrieval and Exposure Analysis

The frozen EgoVLPv2 dual encoder is evaluated on EK100-MIR using mAP and nDCG in both video-to-text (V2T) and text-to-video (T2V) directions.

```bash
bash scripts/02_run_zero_shot_retrieval.sh configs/retrieval.yaml
```

This analysis includes SC/UC/UA comparisons, continuous pair-exposure analysis, PMI analysis, and nested regressions controlling for verb and noun marginal exposure.

### 3. Verb–Noun Binding Diagnostics

Compositional binding is evaluated using controlled noun-swap and verb-swap probes.

```bash
bash scripts/03_run_binding_probe.sh \
  --large-probe <path_to_large_probe_csv> \
  --eval-cache <path_to_eval_cache> \
  --text-embeds <path_to_text_embeds> \
  --checkpoint <path_to_checkpoint> \
  --meta-dir <path_to_ek100_retrieval_annotations> \
  --out-dir outputs/binding
```

The reported diagnostics include noun- and verb-swap accuracies, similarity margins, and margin–exposure correlations.

### 4. Diagnostic Interventions

Retrieval-oriented interventions test whether downstream adaptation improves verb–noun binding in addition to retrieval performance.

```bash
bash scripts/04_run_adapter_interventions.sh
```

The intervention chain includes domain adaptation, verb-focused hard negatives, noun-margin protection, and exposure decorrelation.

### 5. Exposure Reweighting and Counterfactual Verification

Exposure-conditioned score reweighting is applied while keeping video and text representations frozen.

```bash
bash scripts/05_run_counterfactual_exposure.sh \
  --config configs/counterfactual.yaml \
  --egovlpv2-root <path_to_EgoVLPv2> \
  --eval-cache <path_to_eval_cache> \
  --meta-dir <path_to_ek100_retrieval_annotations> \
  --v2t-exposure-csv <path_to_v2t_exposure_csv> \
  --checkpoint-root <path_to_counterfactual_checkpoints> \
  --output-dir outputs/counterfactual
```

Three inference-time exposure conditions are compared:

- **Original**: true sample-level exposure;
- **Zeroed**: exposure information removed;
- **Shuffled**: exposure distribution preserved while sample–exposure correspondence is broken.

## Results

Machine-readable result tables and released dissertation figures are stored under:

```text
results/
├── figures/
└── tables/
```

The CSV files contain the numerical results underlying the corresponding dissertation tables; formatting and rounding may differ from the typeset LaTeX tables in the dissertation.

The mapping between dissertation figures/tables, scripts, and output artefacts is documented in [`docs/experiment_mapping.md`](docs/experiment_mapping.md).

Released and excluded research artefacts are documented in [`docs/artifact_manifest.md`](docs/artifact_manifest.md).

Additional documentation on freezing the Exposure Ledger is provided in [`docs/exposure_freeze.md`](docs/exposure_freeze.md).

## Citation

Citation information will be added upon paper release.

## Acknowledgements

This work builds upon EgoVLPv2, EgoClip, and EPIC-KITCHENS-100. We thank the authors and maintainers of these projects and datasets.
