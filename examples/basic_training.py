#!/usr/bin/env python3
"""
Basic Training Example for CRAFT-DF

This script demonstrates the simplest way to train a CRAFT-DF model
with default settings. Perfect for getting started quickly.

Usage:
    python examples/basic_training.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from craft_df.utils.config import load_default_config
from craft_df.utils.reproducibility import seed_everything
from craft_df.training.training_pipeline import TrainingPipeline


def main():
    """Run basic training example."""
    print("CRAFT-DF Basic Training Example")
    print("=" * 40)
    
    # Load default configuration
    config = load_default_config()
    
    # Set up reproducibility
    seed_everything(config.reproducibility.seed)
    print(f"Set random seed to: {config.reproducibility.seed}")
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(config)
    print("Initialized training pipeline")
    
    # Setup components
    print("Setting up data loaders...")
    train_loader, val_loader, test_loader = pipeline.setup_data_loaders()
    
    print("Setting up model...")
    model = pipeline.setup_model()
    
    print("Setting up trainer...")
    trainer = pipeline.setup_trainer()
    
    # Display model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Max epochs: {config.training.max_epochs}")
    
    # Start training
    print(f"\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    print("\nRunning final evaluation...")
    trainer.test(model, test_loader)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()