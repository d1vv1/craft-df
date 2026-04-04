#!/usr/bin/env python3
"""
Custom Configuration Example for CRAFT-DF

This script demonstrates how to create and use custom configurations
for different training scenarios and hyperparameter experiments.

Usage:
    python examples/custom_configuration.py
"""

import sys
from pathlib import Path
import yaml
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from craft_df.utils.config import load_config, validate_config
from craft_df.utils.reproducibility import seed_everything
from craft_df.training.training_pipeline import TrainingPipeline


def create_custom_config() -> Dict[str, Any]:
    """Create a custom configuration for experimentation."""
    
    custom_config = {
        # Model configuration for smaller memory footprint
        'model': {
            'spatial_backbone': 'mobilenet_v2',
            'spatial_pretrained': True,
            'spatial_freeze_layers': 15,  # Freeze more layers
            'freq_dwt_levels': 2,         # Reduce DWT levels
            'freq_wavelet': 'db4',
            'attention_heads': 4,         # Fewer attention heads
            'attention_dim': 256,         # Smaller attention dimension
            'dropout_rate': 0.2           # Higher dropout for regularization
        },
        
        # Training configuration for faster convergence
        'training': {
            'learning_rate': 2e-4,        # Higher learning rate
            'batch_size': 16,             # Smaller batch size
            'max_epochs': 50,             # Fewer epochs for testing
            'num_workers': 2,
            'pin_memory': True,
            'gradient_clip_val': 0.5,     # Tighter gradient clipping
            'accumulate_grad_batches': 4, # Gradient accumulation
            'precision': 16               # Mixed precision
        },
        
        # Data configuration
        'data': {
            'input_size': [224, 224],
            'dwt_levels': 2,              # Match model config
            'wavelet_type': 'db4',
            'face_confidence_threshold': 0.6,  # Higher confidence
            'train_split': 0.8,           # More training data
            'val_split': 0.1,
            'test_split': 0.1
        },
        
        # Logging configuration
        'logging': {
            'project_name': 'craft-df-custom',
            'experiment_name': 'memory_optimized',
            'log_every_n_steps': 25,     # More frequent logging
            'save_top_k': 5,             # Save more checkpoints
            'monitor': 'val_accuracy',
            'mode': 'max'
        },
        
        # Reproducibility
        'reproducibility': {
            'seed': 123,                 # Different seed
            'deterministic': True,
            'benchmark': False
        },
        
        # Hardware optimization
        'hardware': {
            'accelerator': 'gpu',
            'devices': 1,
            'strategy': 'auto',
            'sync_batchnorm': False
        }
    }
    
    return custom_config


def save_config_to_file(config: Dict[str, Any], filepath: str) -> None:
    """Save configuration to YAML file."""
    
    config_path = Path(filepath)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to: {config_path}")


def main():
    """Run custom configuration example."""
    print("CRAFT-DF Custom Configuration Example")
    print("=" * 45)
    
    # Create custom configuration
    print("Creating custom configuration...")
    custom_config = create_custom_config()
    
    # Save configuration to file
    config_path = "examples/configs/memory_optimized.yaml"
    save_config_to_file(custom_config, config_path)
    
    # Load and validate the configuration
    print("Loading and validating configuration...")
    config = load_config(config_path)
    
    # Validate configuration (this would catch any issues)
    try:
        validate_config(config)
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return
    
    # Set up reproducibility with custom seed
    seed_everything(config.reproducibility.seed)
    print(f"Set random seed to: {config.reproducibility.seed}")
    
    # Display key configuration differences
    print(f"\nKey Configuration Settings:")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Attention heads: {config.model.attention_heads}")
    print(f"Attention dim: {config.model.attention_dim}")
    print(f"DWT levels: {config.model.freq_dwt_levels}")
    print(f"Gradient accumulation: {config.training.accumulate_grad_batches}")
    print(f"Mixed precision: {config.training.precision == 16}")
    
    # Initialize training pipeline with custom config
    print("\nInitializing training pipeline with custom configuration...")
    pipeline = TrainingPipeline(config)
    
    # Setup model to show parameter count
    model = pipeline.setup_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel with Custom Configuration:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Memory footprint: ~{total_params * 4 / 1024**2:.1f} MB (FP32)")
    
    # Show effective batch size with gradient accumulation
    effective_batch_size = config.training.batch_size * config.training.accumulate_grad_batches
    print(f"Effective batch size: {effective_batch_size}")
    
    print("\nCustom configuration setup completed successfully!")
    print(f"You can now train with: python train.py --config {config_path}")


if __name__ == "__main__":
    main()