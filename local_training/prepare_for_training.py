#!/usr/bin/env python3
"""
prepare_for_training.py

Runs all pre-training steps locally:
  1. Kaggle credentials setup
  2. Dataset download  (xhlulu/140k-real-and-fake-faces)
  3. Image organisation  (real/ fake/ split)
  4. Face detection + DWT preprocessing  (via data_prep.py)
  5. Training config generation
  6. Packaging output for Google Drive upload

Output layout (everything you need to upload):
  gdrive_upload/
  ├── processed_data/          ← face crops + DWT .npy files
  ├── metadata.csv             ← dataset index for training
  ├── configs/colab.yaml       ← training config (paths already set for Colab)
  └── processing_summary.json  ← stats / sanity check

Usage:
  python prepare_for_training.py
  python prepare_for_training.py --sample 5000   # quick smoke-test with N images
  python prepare_for_training.py --resume        # continue interrupted preprocessing
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).parent.resolve()
RAW_DATA_DIR   = PROJECT_ROOT / "data" / "raw"          # kaggle download target
IMAGES_DIR     = PROJECT_ROOT / "data" / "images"       # organised real/fake
GDRIVE_DIR     = PROJECT_ROOT / "gdrive_upload"         # final upload bundle
PROCESSED_DIR  = GDRIVE_DIR  / "processed_data"
METADATA_CSV   = GDRIVE_DIR  / "metadata.csv"
CONFIGS_DIR    = GDRIVE_DIR  / "configs"
LOG_FILE       = GDRIVE_DIR  / "prep.log"

# Kaggle dataset identifier
KAGGLE_DATASET = "xhlulu/140k-real-and-fake-faces"

# Expected sub-path after unzip
KAGGLE_REAL = RAW_DATA_DIR / "real_vs_fake" / "real-vs-fake" / "train" / "real"
KAGGLE_FAKE = RAW_DATA_DIR / "real_vs_fake" / "real-vs-fake" / "train" / "fake"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    GDRIVE_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("prep")


# ---------------------------------------------------------------------------
# Step 1 – Kaggle credentials
# ---------------------------------------------------------------------------
def setup_kaggle(log: logging.Logger) -> None:
    log.info("=== Step 1: Kaggle credentials ===")

    # Load .env from project root (no-op if already set in environment)
    load_dotenv(PROJECT_ROOT / ".env")

    username = os.getenv("KAGGLE_USERNAME")
    key      = os.getenv("KAGGLE_KEY")

    if not username or not key:
        raise EnvironmentError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in your .env file or environment.\n"
            "Add them to .env:\n"
            "  KAGGLE_USERNAME=your_username\n"
            "  KAGGLE_KEY=your_api_key"
        )

    kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_cfg.parent.mkdir(exist_ok=True)
    kaggle_cfg.write_text(json.dumps({"username": username, "key": key}))
    if os.name != "nt":
        os.chmod(kaggle_cfg, 0o600)

    log.info(f"Kaggle configured for user: {username}")


# ---------------------------------------------------------------------------
# Step 2 – Download dataset
# ---------------------------------------------------------------------------
def download_dataset(log: logging.Logger) -> None:
    log.info("=== Step 2: Download dataset ===")

    if KAGGLE_REAL.exists() and any(KAGGLE_REAL.iterdir()):
        log.info("Dataset already downloaded — skipping.")
        return

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {KAGGLE_DATASET} (~4 GB) …")
    _run([_kaggle_bin(), "datasets", "download", "-d", KAGGLE_DATASET,
          "-p", str(RAW_DATA_DIR), "--unzip"], log)
    log.info("Download complete.")


# ---------------------------------------------------------------------------
# Step 3 – Organise images
# ---------------------------------------------------------------------------
def organise_images(log: logging.Logger, sample: int = 0) -> None:
    log.info("=== Step 3: Organise images ===")

    real_dst = IMAGES_DIR / "real"
    fake_dst = IMAGES_DIR / "fake"

    if real_dst.exists() and any(real_dst.iterdir()):
        log.info("Images already organised — skipping.")
        return

    for src, dst in [(KAGGLE_REAL, real_dst), (KAGGLE_FAKE, fake_dst)]:
        if not src.exists():
            raise FileNotFoundError(
                f"Expected dataset path not found: {src}\n"
                "Check that the Kaggle download extracted correctly."
            )
        dst.mkdir(parents=True, exist_ok=True)

        files = sorted(src.iterdir())
        if sample:
            files = files[:sample // 2]   # half real, half fake

        log.info(f"Copying {len(files)} images → {dst}")
        for f in files:
            shutil.copy2(f, dst / f.name)

    real_n = sum(1 for _ in real_dst.iterdir())
    fake_n = sum(1 for _ in fake_dst.iterdir())
    log.info(f"Organised: {real_n} real, {fake_n} fake")


# ---------------------------------------------------------------------------
# Step 4 – Preprocessing (face detection + DWT)
# ---------------------------------------------------------------------------
def run_preprocessing(log: logging.Logger, resume: bool = False) -> None:
    log.info("=== Step 4: Face detection + DWT preprocessing ===")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "data_prep.py",
        "--input_dir",   str(IMAGES_DIR),
        "--output_dir",  str(PROCESSED_DIR),
        "--metadata_path", str(METADATA_CSV),
        # images → frame_skip irrelevant, but keep at 1
        "--frame_skip",  "1",
        "--min_detection_confidence", "0.5",
        "--log_level",   "INFO",
        "--log_file",    str(GDRIVE_DIR / "data_prep.log"),
    ]
    if resume:
        cmd.append("--resume")

    _run(cmd, log)

    # Copy processing summary into upload bundle
    summary_src = PROCESSED_DIR / "processing_summary.json"
    if summary_src.exists():
        shutil.copy2(summary_src, GDRIVE_DIR / "processing_summary.json")

    log.info("Preprocessing complete.")


# ---------------------------------------------------------------------------
# Step 5 – Generate Colab training config
# ---------------------------------------------------------------------------
def generate_colab_config(log: logging.Logger) -> None:
    log.info("=== Step 5: Generate Colab training config ===")

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Colab paths assume you mount Drive at /content/drive/MyDrive/craft_df/
    colab_root     = "/content/drive/MyDrive/craft_df"
    colab_metadata = f"{colab_root}/metadata.csv"
    colab_data     = f"{colab_root}/processed_data"

    config = {
        "model": {
            "spatial_backbone":    "mobilenet_v2",
            "spatial_pretrained":  True,
            "spatial_freeze_layers": 10,
            "freq_dwt_levels":     3,
            "freq_wavelet":        "db4",
            "attention_heads":     8,
            "attention_dim":       512,
            "dropout_rate":        0.1,
        },
        "training": {
            "learning_rate":           1e-4,
            "batch_size":              32,
            "max_epochs":              30,
            "num_workers":             2,
            "pin_memory":              True,
            "gradient_clip_val":       1.0,
            "accumulate_grad_batches": 1,
            "precision":               16,
        },
        "data": {
            "input_size":                [224, 224],
            "dwt_levels":                3,
            "wavelet_type":              "db4",
            "face_confidence_threshold": 0.5,
            "train_split":               0.7,
            "val_split":                 0.15,
            "test_split":                0.15,
            "metadata_path":             colab_metadata,
            "data_root":                 colab_data,
        },
        "logging": {
            "project_name":     "craft-df",
            "experiment_name":  "colab-run-1",
            "log_every_n_steps": 10,
            "save_top_k":        3,
            "monitor":           "val/accuracy",
            "mode":              "max",
        },
        "reproducibility": {
            "seed":          42,
            "deterministic": True,
            "benchmark":     False,
        },
        "hardware": {
            "accelerator":    "gpu",
            "devices":        1,
            "strategy":       "auto",
            "sync_batchnorm": False,
        },
    }

    config_path = CONFIGS_DIR / "colab.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    log.info(f"Colab config saved → {config_path}")


# ---------------------------------------------------------------------------
# Step 6 – Summary
# ---------------------------------------------------------------------------
def print_summary(log: logging.Logger) -> None:
    log.info("=== Done ===")

    # Count outputs
    npy_files = list(PROCESSED_DIR.rglob("*.npy")) if PROCESSED_DIR.exists() else []
    face_files = [f for f in npy_files if "_dwt" not in f.name]
    dwt_files  = [f for f in npy_files if "_dwt"  in  f.name]

    log.info(f"Face crops  : {len(face_files)}")
    log.info(f"DWT features: {len(dwt_files)}")
    log.info(f"Metadata CSV: {METADATA_CSV}")
    log.info("")
    log.info("Upload the following folder to Google Drive:")
    log.info(f"  {GDRIVE_DIR}")
    log.info("")
    log.info("In Colab, mount Drive and point your config to:")
    log.info("  /content/drive/MyDrive/craft_df/")
    log.info("  (rename the uploaded folder to 'craft_df' on Drive)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _kaggle_bin() -> str:
    """Return the kaggle CLI path, preferring the active venv's bin."""
    venv = os.getenv("VIRTUAL_ENV")
    if venv:
        candidate = Path(venv) / "bin" / "kaggle"
        if candidate.exists():
            return str(candidate)
    # Fall back to the same bin dir as the running python interpreter
    candidate = Path(sys.executable).parent / "kaggle"
    if candidate.exists():
        return str(candidate)
    return "kaggle"  # last resort: hope it's on PATH


def _run(cmd: list, log: logging.Logger) -> None:
    log.info("$ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="CRAFT-DF local pre-training pipeline")
    parser.add_argument("--sample", type=int, default=0,
                        help="Only process N images (half real, half fake). 0 = all.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted preprocessing step.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download (dataset already present).")
    args = parser.parse_args()

    log = setup_logging()
    log.info("CRAFT-DF  —  local pre-training pipeline")
    log.info(f"Output bundle: {GDRIVE_DIR}")

    if not args.skip_download:
        setup_kaggle(log)
        download_dataset(log)

    organise_images(log, sample=args.sample)
    run_preprocessing(log, resume=args.resume)
    generate_colab_config(log)
    print_summary(log)


if __name__ == "__main__":
    main()
