#!/usr/bin/env python3
"""
Quick training test script - verifies the pipeline works without full data.

Creates a minimal synthetic dataset and runs 1 epoch to test:
- Model initialization
- Data loading
- Forward/backward pass
- Checkpointing

Usage:
  python test_training.py
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure craft_df is importable
sys.path.insert(0, str(Path(__file__).parent))

print("Creating synthetic test dataset...")

# Create temp directory for test data
test_dir = Path("test_data_temp")
test_dir.mkdir(exist_ok=True)

processed_dir = test_dir / "processed"
processed_dir.mkdir(exist_ok=True)

# Create synthetic data (10 real, 10 fake samples)
metadata_records = []

for label, label_name in [(0, "real"), (1, "fake")]:
    label_dir = processed_dir / label_name / "video_001"
    label_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(10):
        # Synthetic face crop (224x224x3 uint8)
        face_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        face_path = label_dir / f"frame_{i:06d}_face_00.npy"
        np.save(face_path, face_crop)
        
        # Synthetic DWT features (200-dim float32 vector)
        dwt_features = np.random.randn(200).astype(np.float32)
        dwt_path = label_dir / f"frame_{i:06d}_face_00_dwt.npy"
        np.save(dwt_path, dwt_features)
        
        # Metadata record
        metadata_records.append({
            "video_id": "video_001",
            "label": label_name,
            "label_numeric": label,
            "frame_number": i,
            "face_index": 0,
            "face_confidence": 0.95,
            "face_path": str(face_path.relative_to(processed_dir)),
            "dwt_path": str(dwt_path.relative_to(processed_dir)),
            "dwt_feature_count": 200,
            "processing_timestamp": "2026-04-06T00:00:00",
            "face_shape": "224x224x3",
            "dwt_wavelet": "db4",
            "dwt_levels": 3,
            "dwt_mode": "symmetric"
        })

# Save metadata CSV
metadata_csv = test_dir / "metadata.csv"
pd.DataFrame(metadata_records).to_csv(metadata_csv, index=False)

print(f"✓ Created {len(metadata_records)} synthetic samples")
print(f"✓ Metadata: {metadata_csv}")

# Create minimal test config
config_dir = test_dir / "configs"
config_dir.mkdir(exist_ok=True)

test_config = {
    "model": {
        "spatial_backbone": "mobilenet_v2",
        "spatial_pretrained": False,  # Faster init without pretrained weights
        "spatial_freeze_layers": 0,
        "freq_dwt_levels": 3,
        "freq_wavelet": "db4",
        "attention_heads": 4,  # Reduced for speed
        "attention_dim": 256,  # Reduced for speed
        "dropout_rate": 0.1,
        "adversarial_training": False,  # Disable for quick test
    },
    "training": {
        "learning_rate": 1e-3,
        "batch_size": 4,  # Small batch
        "max_epochs": 2,  # Just 2 epochs
        "num_workers": 0,  # No multiprocessing for simplicity
        "pin_memory": False,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
        "precision": 32,  # FP32 for CPU compatibility
    },
    "data": {
        "input_size": [224, 224],
        "dwt_levels": 3,
        "wavelet_type": "db4",
        "face_confidence_threshold": 0.5,
        "train_split": 0.6,  # 12 samples
        "val_split": 0.2,    # 4 samples
        "test_split": 0.2,   # 4 samples
        "metadata_path": str(metadata_csv),
        "data_root": str(processed_dir),
    },
    "logging": {
        "project_name": "craft-df-test",
        "experiment_name": "quick-test",
        "log_every_n_steps": 1,
        "save_top_k": 1,
        "monitor": "val/accuracy",
        "mode": "max",
    },
    "reproducibility": {
        "seed": 42,
        "deterministic": False,  # Faster without determinism
        "benchmark": False,
    },
    "hardware": {
        "accelerator": "cpu",  # Force CPU for laptop test
        "devices": 1,
        "strategy": "auto",
        "sync_batchnorm": False,
    },
}

import yaml
config_path = config_dir / "test.yaml"
with open(config_path, "w") as f:
    yaml.dump(test_config, f, default_flow_style=False, sort_keys=False)

print(f"✓ Config: {config_path}")
print("\nStarting training test...\n")

# Run training
from craft_df.training.training_pipeline import TrainingPipeline

try:
    pipeline = TrainingPipeline(
        config_path=str(config_path.resolve()),  # Use absolute path
        experiment_name="laptop_test",
        project_name="craft-df-test",
        debug_mode=True,  # Enable debug mode
        offline_mode=True  # No W&B
    )
    
    results = pipeline.train()
    
    print("\n" + "="*60)
    print("✓ TRAINING TEST PASSED")
    print("="*60)
    print(f"Experiment: {results['experiment_name']}")
    print(f"Training time: {results['training_time_formatted']}")
    print(f"Final epoch: {results['final_epoch']}")
    print(f"Test accuracy: {results['test_results'].get('test_accuracy', 'N/A')}")
    print(f"Best model: {results['best_model_path']}")
    print("\nYour training pipeline is working correctly!")
    
except Exception as e:
    print("\n" + "="*60)
    print("✗ TRAINING TEST FAILED")
    print("="*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
