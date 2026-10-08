"""
Central configuration for PLASyR-Q (hybrid quantum-classical PLASyR follow-up).

Run everything with the interpreter below — the bare `python`/`python3` on
PATH resolves to a broken custom build missing `_ctypes`, and the venv's own
`python`/`python3` symlinks are mismatched to a python3.11 with no packages
installed. `python3.10` inside this venv is the only working, fully-stocked
interpreter (torch 2.5.1+cu121, pennylane 0.42, captum, timm, scikit-learn).
"""
import os

PYTHON_BIN = "/home/vsammartino/progetto_analisi/venv/bin/python3.10"

PROJECT_ROOT = "/home/vsammartino/plasyr-quantum"
RAW_DATA_ROOT = "/home/vsammartino/replay_attacks"

# ── Genuine / replay labeling ────────────────────────────────────────────────
# Per the user: "in the folder replay attack there are folders of spoofed
# satellites and original (in the original folder)".
#   - <RAW_DATA_ROOT>/<Satellite>/...          -> REPLAY  (terrestrial re-emission, 2024)
#   - <RAW_DATA_ROOT>/original/original/...    -> GENUINE (orbital downlink, 2026)
# Confirmed visually at the preprocessing checkpoint (spectrogram + t-SNE,
# matching paper Fig. 6 separability). Flip here if that check disagrees.
REPLAY_IS_PER_SATELLITE_FOLDER = True
GENUINE_SUBDIR = "original/original"

# Satellites with BOTH a genuine (original/) and a replay (per-satellite
# folder) capture ready on disk (no zip extraction needed). This is the
# "reduced" benchmark set — smaller than the paper's 10 satellites because
# Gaofen3-03 has no genuine capture here, and Gaofen12-03/-04 are still
# zipped (Gaofen12-03.zip, Gaofen12-04.zip; left compressed to conserve the
# ~29 GB of free disk on this shared machine).
REDUCED_SATELLITES = [
    "Gaofen12", "Gaofen12-02", "Gaofen3", "RCM1", "RCM3", "Sentinel1A",
]
# Replay-only satellite (no genuine pass available) — usable as an extra
# impostor class in OvR, excluded from replay-detection.
REPLAY_ONLY_SATELLITES = ["Gaofen3-03"]

# ── IQ format ─────────────────────────────────────────────────────────────────
IQ_DTYPE = "complex64"  # interleaved float32 I/Q, no header (verified by hand)

# Cap raw samples read per file during preprocessing. Files are 0.7-5.2 GB;
# reading the full file is unnecessary (SAR illuminating-signal statistics
# are stationary across a pass) and would blow the disk/time budget for the
# reduced run. ~200M complex64 samples = 1.6 GB read, a few seconds.
MAX_RAW_SAMPLES_PER_FILE = 200_000_000

# ── Magnitude thresholding (Eq. 2 in the paper; pre_processingBurst.py) ───────
ALPHA = 0.6

# ── Burst segmentation (pre_processingBurst.py defaults) ─────────────────────
BURST_WINDOW = 50_000
BURST_T_MIN = 500
BURST_ENERGY_THRESH_FACTOR = 0.05  # repo's operational value (paper text says ~1%)
MIN_BURST_SAMPLES_FOR_IMAGE = 64 * 1024  # need >= NUM_STACKS*FFT_SIZE samples

# ── Spectrogram generation (iqToSpectrograms.py) ──────────────────────────────
FFT_SIZE = 1024
NUM_STACKS = 64
IMAGE_SIZE = (224, 224)

# Cap images written per raw file, to bound disk usage on a ~29 GB-free disk.
# A 224x224 8-bit grayscale PNG is a few KB; this caps the reduced dataset to
# a manageable, still-statistically-useful size.
MAX_IMAGES_PER_FILE = 120

SPECTROGRAM_ROOT = os.path.join(PROJECT_ROOT, "data_processed")

# ── Dataset / split ───────────────────────────────────────────────────────────
# Pass-level split where possible (>=2 passes/class); otherwise temporal
# split *within* the single available pass (first 60% of its images -> train,
# next 20% -> val, last 20% -> test). Never image-shuffle across a pass.
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2
SPLIT_SEED = 42

# ── Quantum branch (VQC) ──────────────────────────────────────────────────────
N_QUBITS = 6
VQC_LAYERS = 3
QUANTUM_DEVICE = "default.qubit"   # backprop on default.qubit beat lightning.qubit
QUANTUM_DIFF_METHOD = "backprop"   # (adjoint/parameter-shift) in our GPU sanity check

# Optimized hybrid config: wider/deeper VQC + separate (higher) LR for the
# quantum branch. VQC rotation-gate gradients are small in magnitude relative
# to the pretrained CNN backbone's, so a shared LR under-trains the quantum
# branch; a dedicated higher LR lets it actually converge within the epoch
# budget. Gradient clipping keeps this higher LR from destabilizing training.
N_QUBITS_OPT = 8
VQC_LAYERS_OPT = 4
QUANTUM_LR = 5e-3
GRAD_CLIP_NORM = 1.0

# ── Backbones (reduced benchmark) ─────────────────────────────────────────────
BACKBONES = ["se_resnet34", "densenet121"]

# ── Fusion operators ───────────────────────────────────────────────────────────
FUSION_OPS = ["concat_linear", "sum", "product", "weighted", "gated", "max"]

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS_SMOKE = 8
EPOCHS_REDUCED = 40
NUM_WORKERS = 4
SEED = 42

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
