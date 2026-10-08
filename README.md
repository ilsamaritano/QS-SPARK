# QSpark: Quantum SAR Physical-Layer Authentication with Replay-Attack Detection

Code and result tables for the paper *QSpark: Quantum SAR Physical-Layer
Authentication with Replay-Attack Detection*.

QSpark is a hybrid quantum–classical replay detector for synthetic aperture
radar (SAR) satellites. A CNN backbone turns a burst spectrogram into a
feature vector `f`; a 6-qubit, 54-parameter variational quantum circuit (VQC)
reads `f` and returns six Pauli-Z expectation values `<Z>`. The two branches
are fused at the logit level for training, and replay detection scores the
joint embedding `[f; <Z>]` with a Mahalanobis anomaly model fit on genuine
data only. The headline metric is the true replay rejection rate at a 1%
false-alarm rate (TRR@FAR 1%).

## Repository layout

```
config.py                  All paths and hyperparameters (edit this first)
preprocessing/
  pipeline.py              Thresholding, burst segmentation, STFT, 224x224 images
  run_preprocessing.py     Raw IQ captures -> data_processed/<class>/<sat>/<pass>/
data/dataset.py            Image discovery, leakage-free splits, OvR datasets
models/
  backbones.py             SE-ResNet-34, DenseNet-121, and any timm backbone
  vqc.py                   Angle embedding + strongly entangling layers (PennyLane)
  fusion.py                concat_linear, sum, weighted, product, gated, max
  hybrid.py                HybridModel, ClassicalOnlyModel, checkpoint loading
train/
  train_ovr.py             One-vs-rest training of a single configuration
  benchmark.py             Train + evaluate a (backbone x fusion x satellite) matrix
  smoke_e2e.py             Short end-to-end sanity run
eval/
  metrics.py               AUROC, EER, TRR@FAR, TAR@FAR, GaussianMahalanobis
  replay_eval.py           Replay detection (Mahalanobis on the joint embedding)
  ovr_eval.py              One-vs-rest authentication
  adaptive_pgd.py          White-box PGD on the differentiable Mahalanobis score
  merge_pgd_results.py     Merge PGD sweeps run in separate processes
explain/
  gradcam.py               Grad-CAM on the final convolutional block
  quantum_attr.py          Per-qubit <Z> and parameter-shift sensitivity
  run_explain_matrix.py    Run both over a set of checkpoints
paper/                     Table and figure scripts, LaTeX source, figures/
scratch/                   Shell drivers used for the reported runs (two GPUs)
data_processed/            Spectrogram dataset: 4,320 PNGs (1,080 genuine, 3,240 replay)
results/                   Main benchmark: SE-ResNet-34 and DenseNet-121, 7 models each
results_backbones/         Backbone benchmark: five further architectures
results_optimized_v2/      Ablation: 8 qubits, 4 layers, separate quantum learning rate
results_optimized/         Discarded 60-epoch pilot of the same ablation
```

## Requirements

Python 3.10 and a CUDA GPU. The reported runs used two NVIDIA RTX A6000
cards; one GPU is enough if you drop the `CUDA_VISIBLE_DEVICES` split.

| Package | Version used |
|---|---|
| torch / torchvision | 2.5.1 (cu121) / 0.20.1 |
| pennylane | 0.42.3 |
| timm | 1.0.27 |
| captum | 0.9.0 |
| scikit-learn | 1.7.2 |
| numpy / scipy / pandas | 2.1.2 / 1.15.3 / 2.3.0 |
| matplotlib / pillow | 3.10.9 / 11.0.0 |

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install pennylane==0.42.3 timm==1.0.27 captum==0.9.0 scikit-learn pandas scipy matplotlib pillow
```

## Setup

Paths are absolute and set in `config.py`. Before running anything, edit:

- `PROJECT_ROOT`: the directory containing this README. `SPECTROGRAM_ROOT`
  and `RESULTS_ROOT` are derived from it.
- `RAW_DATA_ROOT`: the raw IQ captures. Only needed for preprocessing.

The scripts in `scratch/` hardcode the same paths plus an interpreter and a
log directory (`ROOT`, `PY`, `LOGDIR` at the top of each file). Set those
too, or copy the commands below instead.

All commands are run from the repository root.

## Data

`data_processed/` holds the extracted spectrogram dataset used in the paper:

| Satellite | Genuine passes | Replay passes | Genuine images | Replay images |
|---|---|---|---|---|
| Gaofen12 | 2 | 5 | 240 | 600 |
| Gaofen12-02 | 1 | 4 | 120 | 480 |
| Gaofen3 | 2 | 5 | 240 | 600 |
| RCM1 | 1 | 3 | 120 | 360 |
| RCM3 | 1 | 5 | 120 | 600 |
| Sentinel1A | 2 | 5 | 240 | 600 |

With this directory in place you can skip preprocessing. The raw IQ captures
(about 79 GB) are not included.

To rebuild the dataset from raw captures, `RAW_DATA_ROOT` must contain
headerless interleaved-float32 `complex64` files laid out as:

```
<RAW_DATA_ROOT>/<Satellite>/<pass file>              replay captures
<RAW_DATA_ROOT>/original/original/<Satellite>_...    genuine captures
```

```bash
python preprocessing/run_preprocessing.py --max_images_per_file 120
```

The chain is magnitude thresholding (`ALPHA = 0.6`), burst segmentation
(window 50,000 samples, minimum burst 500), per-burst mean-power
normalization, and a 1024-point STFT stacked 64 frames deep into a 224x224
grayscale image. Each raw file is capped at 200M samples and 120 images.

### Splits

No acquisition pass appears in two splits. A satellite with two genuine
passes trains and validates on one (temporal 75/25) and tests on the other.
A satellite with one genuine pass uses a temporal 60/20/20 split inside it.
The split is reconstructed from the seed stored in each checkpoint.

## Reproducing the results

### 0. Sanity check

```bash
python train/benchmark.py --epochs 2 --satellites Gaofen12 Gaofen3 --fusions classical sum
```

### 1. Main benchmark (84 configurations)

Two backbones x {classical + six fusion operators} x six satellites, 40
epochs, seed 42. The defaults in `config.py` are the paper's settings, so
this is equivalent to `scratch/run_overnight.sh` phase 1 on one GPU:

```bash
python train/benchmark.py \
  --backbones se_resnet34 densenet121 \
  --fusions classical concat_linear sum weighted product gated max \
  --epochs 40 --seeds 42 --out_root results
```

`classical` trains the backbone with no VQC. Each configuration writes
`results/ovr/<satellite>/<backbone>__<fusion>/seed42/{best.pt,train_results.json}`,
and the run ends by writing `benchmark_per_sat.csv`, `benchmark_per_sat.json`
and `benchmark_macro.csv` under `--out_root`.

To split the work across GPUs, launch one process per backbone or per
satellite subset, then rebuild the tables from the checkpoints with a single
process. Parallel runs race on the CSV, so this last step is required:

```bash
python train/benchmark.py --backbones se_resnet34 densenet121 --skip_train --out_root results
```

`--skip_train` re-evaluates existing checkpoints without training, which is
also how to regenerate the tables from released checkpoints.

### 2. Backbone benchmark

```bash
python train/benchmark.py \
  --backbones resnet18 resnet34 efficientnet_b0 mobilenetv3 vit_small \
  --fusions classical concat_linear product \
  --epochs 40 --seeds 42 --out_root results_backbones
```

### 3. Quantum-branch ablation

Same protocol, SE-ResNet-34 only, with a larger circuit and a dedicated
quantum learning rate:

```bash
python train/benchmark.py \
  --backbones se_resnet34 --fusions concat_linear sum gated product \
  --epochs 40 --seeds 42 --out_root results_optimized_v2 \
  --n_qubits 8 --n_layers 4 --quantum_lr 5e-3 --grad_clip_norm 1.0
```

### 4. Adaptive white-box attack

20-step L-infinity PGD that minimizes the Mahalanobis score of replayed
spectrograms, with gradients through the backbone, the simulated circuit and
the PCA projection. The threshold is fixed before the attack.

```bash
python eval/adaptive_pgd.py \
  --backbones se_resnet34 densenet121 \
  --fusions classical sum gated product \
  --epsilons 0 0.02 0.05 0.10 0.20 0.30 \
  --pgd_steps 20 --max_attack 100 --seeds 42 \
  --ckpt_root results --out_root results
```

Output goes to `<out_root>/adaptive_pgd/{per_sat_sweep.csv,macro_curve.csv,all_rows.json}`.
Note that the script's own defaults (`--pgd_steps 50 --max_attack 500`)
differ from the values used in the paper. If you ran one process per
backbone with separate `--out_root` directories, combine them:

```bash
python eval/merge_pgd_results.py \
  --inputs results/adaptive_pgd_se results/adaptive_pgd_dn --out_root results
```

### 5. Explainability

```bash
python explain/run_explain_matrix.py \
  --backbone se_resnet34 --fusions classical sum gated product \
  --satellites Gaofen12 Gaofen3 --seed 42 --ckpt_root results
```

Writes Grad-CAM panels, per-qubit `<Z>` plots, parameter-shift importance
plots and gate reports to `results/explain/`. `explain/gradcam.py` and
`explain/quantum_attr.py` can also be run on a single checkpoint with
`--ckpt <path> --target_sat <satellite>`.

### 6. Tables and figures

These read the CSVs and checkpoints produced above.

| Script | Needs GPU | Produces |
|---|---|---|
| `paper/comparative_table.py` | no | Classical vs. best hybrid, per satellite and macro; writes `results/comparative_table_*.csv` |
| `paper/backbone_table.py` | no | Backbone benchmark table, printed with LaTeX rows |
| `paper/extract_figure_data.py` | yes | `paper/fig_data.npz`: Mahalanobis scores and per-qubit `<Z>` values |
| `paper/make_paper_figures.py` | no | Architecture, TRR dumbbell, score distributions, PGD sweep, fusion heatmap, per-qubit `<Z>` |
| `paper/compose_gradcam_figure.py` | no | `fig_gradcam.pdf`, from the PNGs in `results/explain/` |
| `paper/generate_figures.py` | yes | Supplementary heatmaps, per-fusion bars, t-SNE |
| `paper/plot_quantum_vs_classical.py` | yes | Supplementary t-SNE comparisons and score histograms |
| `paper/plot_adaptive_adversary.py` | no | Supplementary per-backbone PGD plots |

Figures are written to `paper/figures/`. `paper/fig_data.npz` is included,
so `make_paper_figures.py` runs without checkpoints or a GPU.

## Where the reported numbers come from

| In the paper | File |
|---|---|
| Per-satellite replay and OvR results, fusion heatmap | `results/benchmark_per_sat.csv` |
| Macro results per fusion operator | `results/benchmark_macro.csv` |
| Classical vs. best hybrid | `results/comparative_table_macro.csv`, `results/comparative_table_per_sat_<backbone>.csv` |
| Backbone benchmark | `results_backbones/benchmark_per_sat.csv` together with `results/benchmark_per_sat.csv` |
| Adaptive PGD sweep | `results/adaptive_pgd/per_sat_sweep.csv`, `results/adaptive_pgd/macro_curve.csv` |
| Enlarged-circuit ablation | `results_optimized_v2/benchmark_per_sat.csv` |
| Gate statistics | `results/explain/gate_report_*.json` |

Columns in `benchmark_per_sat.csv`: `backbone, fusion, satellite, seed,
replay_auroc, replay_eer, replay_trr_far1, ovr_auroc, ovr_eer, ovr_tar_far1`.

## Evaluation protocol as implemented

- **Anomaly model** (`eval/metrics.py`, `GaussianMahalanobis`): a Gaussian is
  fit on the embeddings of the target satellite's genuine training images.
  When the sample count does not exceed the embedding dimension, embeddings
  are first projected onto at most 64 principal components.
- **Replay scoring** (`eval/replay_eval.py`): the satellite's held-out genuine
  images and all of its replay images are scored by squared Mahalanobis
  distance. Classical-only and quantum-only scores are also reported.
- **Threshold**: `trr_at_far` sets the threshold at the 99th percentile of the
  held-out genuine scores and reports the fraction of replay scores at or
  above it.
- **Best hybrid fusion**: `paper/comparative_table.py` and
  `paper/backbone_table.py` pick, per satellite, the fusion operator with the
  highest `replay_trr_far1` in `benchmark_per_sat.csv`.
- **Training** (`train/train_ovr.py`): one model per satellite, one-vs-rest,
  Adam at learning rate 1e-4 with cosine decay, batch 32, 40 epochs, seed 42,
  all parameters trained jointly. The product rule uses a negative
  log-likelihood loss; the other operators use cross-entropy.
- **Quantum circuit** (`models/vqc.py`): PennyLane `default.qubit` with
  backpropagation. All results are from noiseless simulation.

## Notes

- Every reported number comes from a single seed (42).
- Checkpoints total about 14 GB (232 `best.pt` files). If they are not
  distributed with the code, steps 1 to 3 regenerate them.
- `results_optimized/` is the 60-epoch pilot that the paper discards because
  the longer schedule also degraded the classical models. It is kept for
  completeness and is not used by any table.
