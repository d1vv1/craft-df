#!/usr/bin/env python3
"""
CRAFT-DF Training Notebook Script
==================================
Copy each section into a separate Colab/Kaggle cell.
Run cells top to bottom, one at a time.

HOW TO USE:
1. Push your code to GitHub first
2. Open colab.research.google.com (or kaggle.com/code)
3. Create a new notebook
4. Copy each CELL section below into its own notebook cell
5. Run them in order
"""

# =============================================================================
# CELL 1 - Check GPU
# Run this first to confirm you have a GPU available
# =============================================================================
"""
import torch

print("Python and PyTorch are ready!")
print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
else:
    print("No GPU found.")
    print("In Colab: Runtime > Change runtime type > T4 GPU")
    print("In Kaggle: Settings (right panel) > Accelerator > GPU T4 x2")
"""

# =============================================================================
# CELL 2 - Mount Google Drive (Colab only, skip on Kaggle)
# Your checkpoints and processed data will be saved here permanently
# =============================================================================
"""
from google.colab import drive
drive.mount('/content/drive')

import os
# Create a folder in your Drive for this project
PROJECT_DIR = '/content/drive/MyDrive/craft-df-training'
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(f'{PROJECT_DIR}/checkpoints', exist_ok=True)
os.makedirs(f'{PROJECT_DIR}/processed_data', exist_ok=True)

print(f"Project directory ready: {PROJECT_DIR}")
print("Your files will be saved here even after the session ends.")
"""

# =============================================================================
# CELL 3 - Clone your repo and install dependencies
# Replace YOUR_USERNAME with your actual GitHub username
# =============================================================================
"""
import os

# Clone the repo
# IMPORTANT: Replace with your actual GitHub repo URL
GITHUB_REPO = "https://github.com/YOUR_USERNAME/craft-df.git"

!git clone {GITHUB_REPO} /content/craft-df
%cd /content/craft-df

# Install all dependencies
!pip install -q -r requirements.txt

# Verify installation
import craft_df
print("CRAFT-DF installed successfully!")
"""

# =============================================================================
# CELL 4 - Download the dataset (FaceForensics++ small version via Kaggle API)
#
# OPTION A: Use Kaggle dataset (recommended - no download to your laptop)
# OPTION B: Upload your own videos to Google Drive and point to them
# =============================================================================
"""
# --- OPTION A: Download from Kaggle directly inside Colab ---
# First, upload your kaggle.json API key:
#   1. Go to kaggle.com > Account > Create New API Token
#   2. It downloads kaggle.json to your laptop
#   3. Run the cell below to upload it

from google.colab import files
files.upload()  # Upload your kaggle.json here

import os
os.makedirs('/root/.kaggle', exist_ok=True)
!cp kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

# Download a small deepfake dataset (this one is ~2GB, manageable)
!kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p /content/data --unzip

# Check what we got
!ls /content/data/

# --- OPTION B: If you already have videos in Google Drive ---
# Just set DATA_DIR to point to them, e.g.:
# DATA_DIR = '/content/drive/MyDrive/my-videos'
"""

# =============================================================================
# CELL 5 - Organize data into the required structure
#
# CRAFT-DF expects:
#   input_videos/
#   ├── real/   <-- real videos here
#   └── fake/   <-- deepfake videos here
# =============================================================================
"""
import os
import shutil
from pathlib import Path

# Set these paths based on where your data landed
RAW_DATA_DIR = '/content/data'          # Where you downloaded/uploaded data
INPUT_DIR    = '/content/input_videos'  # Where we'll organize it

os.makedirs(f'{INPUT_DIR}/real', exist_ok=True)
os.makedirs(f'{INPUT_DIR}/fake', exist_ok=True)

# --- Adjust this block to match your dataset's folder structure ---
# Example for the 140k-real-and-fake-faces dataset:
real_src = f'{RAW_DATA_DIR}/real_and_fake_face/training_real'
fake_src = f'{RAW_DATA_DIR}/real_and_fake_face/training_fake'

# If your dataset has images instead of videos, that's fine too
# Just copy them into real/ and fake/ folders
if os.path.exists(real_src):
    !cp -r {real_src}/* {INPUT_DIR}/real/
    !cp -r {fake_src}/* {INPUT_DIR}/fake/

# Check counts
real_count = len(list(Path(f'{INPUT_DIR}/real').iterdir()))
fake_count = len(list(Path(f'{INPUT_DIR}/fake').iterdir()))
print(f"Real samples: {real_count}")
print(f"Fake samples: {fake_count}")
print("Data organized successfully!")
"""

# =============================================================================
# CELL 6 - Run data preparation (face detection + DWT processing)
# This converts raw videos/images into the .npy format CRAFT-DF needs
# Takes 10-60 minutes depending on dataset size
# =============================================================================
"""
import os

INPUT_DIR    = '/content/input_videos'
OUTPUT_DIR   = '/content/drive/MyDrive/craft-df-training/processed_data'  # Saved to Drive
METADATA_CSV = '/content/drive/MyDrive/craft-df-training/metadata.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run data preparation
# --frame_skip 5 means process every 5th frame (faster, less data)
# --min_detection_confidence 0.5 is more permissive (detects more faces)
!python data_prep.py \
    --input_dir {INPUT_DIR} \
    --output_dir {OUTPUT_DIR} \
    --metadata_path {METADATA_CSV} \
    --frame_skip 5 \
    --min_detection_confidence 0.5 \
    --max_frames_per_video 100

# Check results
import pandas as pd
if os.path.exists(METADATA_CSV):
    df = pd.read_csv(METADATA_CSV)
    print(f"Total samples processed: {len(df)}")
    print(f"Real: {len(df[df['label']==0])}")
    print(f"Fake: {len(df[df['label']==1])}")
    print(df.head())
"""

# =============================================================================
# CELL 7 - Create training configuration
# Tuned for Colab's T4 GPU (15GB VRAM)
# =============================================================================
"""
import yaml
import os

config = {
    'model': {
        'spatial_backbone': 'mobilenet_v2',
        'spatial_pretrained': True,
        'spatial_freeze_layers': 10,
        'freq_dwt_levels': 3,
        'freq_wavelet': 'db4',
        'attention_heads': 8,
        'attention_dim': 512,
        'dropout_rate': 0.1,
    },
    'training': {
        'learning_rate': 1e-4,
        'batch_size': 32,        # T4 can handle 32 comfortably
        'max_epochs': 30,        # Start with 30, increase if needed
        'num_workers': 2,        # Colab has 2 CPU cores
        'pin_memory': True,
        'gradient_clip_val': 1.0,
        'accumulate_grad_batches': 1,
        'precision': 16,         # Mixed precision = faster + less memory
    },
    'data': {
        'input_size': [224, 224],
        'dwt_levels': 3,
        'wavelet_type': 'db4',
        'face_confidence_threshold': 0.5,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'metadata_path': '/content/drive/MyDrive/craft-df-training/metadata.csv',
    },
    'logging': {
        'project_name': 'craft-df',
        'experiment_name': 'colab-run-1',
        'log_every_n_steps': 10,
        'save_top_k': 3,
        'monitor': 'val_accuracy',
        'mode': 'max',
    },
    'reproducibility': {
        'seed': 42,
        'deterministic': True,
        'benchmark': False,
    },
    'hardware': {
        'accelerator': 'gpu',
        'devices': 1,
        'strategy': 'auto',
        'sync_batchnorm': False,
    },
}

# Save config
os.makedirs('configs', exist_ok=True)
with open('configs/colab.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Config saved to configs/colab.yaml")
print(f"Batch size: {config['training']['batch_size']}")
print(f"Max epochs: {config['training']['max_epochs']}")
print(f"Mixed precision: {config['training']['precision'] == 16}")
"""

# =============================================================================
# CELL 8 - (Optional) Set up Weights & Biases for experiment tracking
# Free at wandb.ai — gives you live loss/accuracy charts
# Skip this cell if you don't want it, training still works without it
# =============================================================================
"""
!pip install -q wandb

import wandb
wandb.login()  # Paste your API key from wandb.ai/authorize when prompted

print("W&B ready. Your training charts will appear at wandb.ai")
"""

# =============================================================================
# CELL 9 - START TRAINING
# This is the main event. Watch the loss go down!
# Expected time on T4 GPU: ~20-40 min for 30 epochs on a small dataset
# =============================================================================
"""
import os

CHECKPOINT_DIR = '/content/drive/MyDrive/craft-df-training/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Run training
!python train.py \
    --config configs/colab.yaml \
    --experiment colab-run-1 \
    --offline  # Remove --offline if you set up W&B in Cell 8

print("Training complete!")
print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
"""

# =============================================================================
# CELL 10 - Check training results
# =============================================================================
"""
import os
from pathlib import Path

CHECKPOINT_DIR = '/content/drive/MyDrive/craft-df-training/checkpoints'

# List saved checkpoints
checkpoints = list(Path(CHECKPOINT_DIR).glob('*.ckpt'))
if checkpoints:
    print(f"Saved {len(checkpoints)} checkpoints:")
    for ckpt in sorted(checkpoints):
        size_mb = ckpt.stat().st_size / 1e6
        print(f"  {ckpt.name}  ({size_mb:.1f} MB)")
else:
    print("No checkpoints found yet.")

# The best checkpoint is what you'll use for inference
best_ckpt = CHECKPOINT_DIR + '/best.ckpt'
if os.path.exists(best_ckpt):
    print(f"\nBest model: {best_ckpt}")
    print("You can now use this for inference!")
"""

# =============================================================================
# CELL 11 - Quick inference test on a sample video
# =============================================================================
"""
import torch
import sys
sys.path.insert(0, '/content/craft-df')

from craft_df.models.craft_df_model import CRAFTDFModel

CHECKPOINT_PATH = '/content/drive/MyDrive/craft-df-training/checkpoints/best.ckpt'

# Load trained model
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
model = CRAFTDFModel()
model.load_state_dict(checkpoint['state_dict'])
model.eval()

print("Model loaded successfully!")
print("Ready for inference on new videos.")

# To run inference on a video:
# from examples.model_inference import CRAFTDFInference
# inference = CRAFTDFInference(CHECKPOINT_PATH)
# result = inference.predict_video('/path/to/video.mp4')
# print(result)
"""

# =============================================================================
# CELL 12 - Resume training if session disconnected
# Colab disconnects after ~12 hours. Use this to pick up where you left off.
# =============================================================================
"""
import os

# Find the last checkpoint
CHECKPOINT_DIR = '/content/drive/MyDrive/craft-df-training/checkpoints'
last_ckpt = CHECKPOINT_DIR + '/last.ckpt'

if os.path.exists(last_ckpt):
    print(f"Resuming from: {last_ckpt}")
    !python train.py \
        --config configs/colab.yaml \
        --experiment colab-run-1-resumed \
        --resume {last_ckpt} \
        --offline
else:
    print("No checkpoint found to resume from. Run Cell 9 first.")
"""
