#!/usr/bin/env python3
"""
CRAFT-DF Training Script

Main training script for the CRAFT-DF deepfake detection model.
Provides a comprehensive command-line interface for training with
experiment tracking, checkpointing, and performance monitoring.

Usage:
    python train.py --config configs/default.yaml --experiment my_experiment
    python train.py --config configs/default.yaml --resume checkpoints/last.ckpt
    python train.py --config configs/default.yaml --debug
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import traceback
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from craft_df.training.training_pipeline import TrainingPipeline
from craft_df.utils.config import load_config, validate_config
from craft_df.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CRAFT-DF Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration file (YAML format)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default=None,
        help='Experiment name for logging and checkpointing'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='craft-df',
        help='Project name for Weights & Biases logging'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with reduced functionality'
    )
    
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Run in offline mode (no W&B sync)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without training'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Override data path from config'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Override number of data loading workers'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override maximum epochs from config'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Override learning rate from config'
    )
    
    parser.add_argument(
        '--precision',
        type=int,
        choices=[16, 32],
        default=None,
        help='Override training precision (16 for mixed precision)'
    )
    
    # Hardware arguments
    parser.add_argument(
        '--gpus',
        type=int,
        default=None,
        help='Number of GPUs to use'
    )
    
    parser.add_argument(
        '--accelerator',
        type=str,
        choices=['cpu', 'gpu', 'tpu'],
        default=None,
        help='Accelerator type'
    )
    
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def override_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override configuration with command line arguments."""
    overridden_config = config.copy()
    
    # Data overrides
    if args.data_path:
        overridden_config.setdefault('data', {})['metadata_path'] = args.data_path
    
    if args.batch_size:
        overridden_config.setdefault('training', {})['batch_size'] = args.batch_size
    
    if args.num_workers:
        overridden_config.setdefault('training', {})['num_workers'] = args.num_workers
    
    # Training overrides
    if args.epochs:
        overridden_config.setdefault('training', {})['max_epochs'] = args.epochs
    
    if args.learning_rate:
        overridden_config.setdefault('training', {})['learning_rate'] = args.learning_rate
    
    if args.precision:
        overridden_config.setdefault('training', {})['precision'] = args.precision
    
    # Hardware overrides
    if args.gpus:
        overridden_config.setdefault('hardware', {})['devices'] = args.gpus
    
    if args.accelerator:
        overridden_config.setdefault('hardware', {})['accelerator'] = args.accelerator
    
    return overridden_config


def validate_environment() -> None:
    """Validate that the environment is properly set up."""
    try:
        import torch
        import pytorch_lightning as pl
        import wandb
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"PyTorch Lightning version: {pl.__version__}")
        logger.info(f"Weights & Biases version: {wandb.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.warning("CUDA not available - training will use CPU")
            
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        raise RuntimeError(f"Environment validation failed: {e}")


def main() -> int:
    """Main training function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.log_level)
        
        logger.info("Starting CRAFT-DF training script")
        logger.info(f"Arguments: {vars(args)}")
        
        # Validate environment
        validate_environment()
        
        # Load and validate configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override config with command line arguments
        config = override_config(config, args)
        
        # Validate configuration
        validate_config(config)
        logger.info("Configuration validation passed")
        
        # If validate-only mode, exit here
        if args.validate_only:
            logger.info("Configuration validation completed - exiting")
            return 0
        
        # Set up reproducibility
        seed = config.get('reproducibility', {}).get('seed', 42)
        seed_everything(seed)
        logger.info(f"Set random seed to: {seed}")
        
        # Initialize training pipeline
        pipeline = TrainingPipeline(
            config_path=args.config,
            experiment_name=args.experiment,
            project_name=args.project,
            resume_from_checkpoint=args.resume,
            debug_mode=args.debug,
            offline_mode=args.offline
        )
        
        # Override pipeline config with merged config
        pipeline.config = config
        
        # Start training
        logger.info("Initializing training pipeline...")
        results = pipeline.train()
        
        # Log final results
        logger.info("Training completed successfully!")
        logger.info(f"Experiment: {results['experiment_name']}")
        logger.info(f"Training time: {results['training_time_formatted']}")
        logger.info(f"Final test accuracy: {results['test_results'].get('test_accuracy', 'N/A')}")
        logger.info(f"Best model: {results['best_model_path']}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)