"""
CRAFT-DF Training Pipeline

This module implements a comprehensive training pipeline for the CRAFT-DF model
with experiment tracking, automatic checkpointing, and proper data management.
The pipeline integrates Weights & Biases for experiment tracking and provides
robust training capabilities with GPU optimization.

Key Features:
- Weights & Biases integration for experiment tracking
- Automatic checkpointing and model recovery
- Train/validation/test data splits
- Configuration management for hyperparameters
- GPU optimization and mixed precision training
- Distributed training support
- Performance monitoring and profiling
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import yaml
import json
import time
from datetime import datetime
import psutil
try:
    import GPUtil
    _GPUTIL_AVAILABLE = True
except ImportError:
    _GPUTIL_AVAILABLE = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    DeviceStatsMonitor, ModelSummary
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler

# CRAFT-DF imports
from ..models.craft_df_model import CRAFTDFModel
from ..data.dataset import HierarchicalDeepfakeDataset
from ..data.transforms import get_transforms
from ..utils.config import load_config, validate_config
from ..utils.reproducibility import seed_everything
from .performance_monitor import PerformanceCallback, PerformanceMonitor, GPUOptimizer, DistributedTrainingOptimizer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Comprehensive training pipeline for CRAFT-DF model.
    
    This class orchestrates the complete training process including:
    - Data loading and preprocessing
    - Model initialization and configuration
    - Training loop with experiment tracking
    - Checkpointing and model recovery
    - Performance monitoring and optimization
    
    Args:
        config_path: Path to configuration file
        experiment_name: Name for the experiment (used in logging)
        project_name: Project name for Weights & Biases
        resume_from_checkpoint: Path to checkpoint to resume from
        debug_mode: Enable debug mode with reduced functionality
        offline_mode: Run without internet connectivity (no W&B sync)
    """
    
    def __init__(
        self,
        config_path: str,
        experiment_name: Optional[str] = None,
        project_name: str = "craft-df",
        resume_from_checkpoint: Optional[str] = None,
        debug_mode: bool = False,
        offline_mode: bool = False
    ):
        self.config_path = Path(config_path)
        self.experiment_name = experiment_name or f"craft_df_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.project_name = project_name
        self.resume_from_checkpoint = resume_from_checkpoint
        self.debug_mode = debug_mode
        self.offline_mode = offline_mode
        
        # Load and validate configuration
        self.config = self._load_and_validate_config()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.model = None
        self.trainer = None
        self.logger_instance = None
        self.data_loaders = {}
        
        # Performance optimization
        self.gpu_optimizer = GPUOptimizer(
            enable_amp=self.config.get('training', {}).get('precision', 32) == 16,
            memory_fraction=0.9,
            allow_growth=True,
            enable_h100_optimizations=True
        )
        
        # Distributed training optimizer
        self.distributed_optimizer = None
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(
            log_interval=self.config.get('logging', {}).get('log_every_n_steps', 50),
            save_interval=1000,
            enable_detailed_profiling=not self.debug_mode
        )
        
        # Performance tracking
        self.training_start_time = None
        self.performance_metrics = {}
        
        logger.info(f"TrainingPipeline initialized for experiment: {self.experiment_name}")
        logger.info(f"Configuration loaded from: {self.config_path}")
        logger.info(f"Debug mode: {self.debug_mode}, Offline mode: {self.offline_mode}")
    
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load and validate configuration file."""
        try:
            config = load_config(str(self.config_path))
            validate_config(config)
            
            # Apply debug mode modifications
            if self.debug_mode:
                config = self._apply_debug_config(config)
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise RuntimeError(f"Configuration loading failed: {e}")
    
    def _apply_debug_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply debug mode modifications to configuration."""
        debug_config = config.copy()
        
        # Reduce training parameters for faster debugging
        debug_config['training']['max_epochs'] = min(3, config['training']['max_epochs'])
        debug_config['training']['batch_size'] = min(4, config['training']['batch_size'])
        debug_config['logging']['log_every_n_steps'] = 1
        
        # Disable some expensive features
        debug_config['training']['precision'] = 32  # Disable mixed precision
        debug_config['hardware']['strategy'] = 'auto'  # Disable distributed training
        
        logger.info("Applied debug mode configuration modifications")
        return debug_config
    
    def _setup_logging(self) -> None:
        """Setup comprehensive logging configuration."""
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'training_{self.experiment_name}.log')
            ]
        )
        
        # Set specific logger levels
        logging.getLogger('pytorch_lightning').setLevel(logging.INFO)
        logging.getLogger('wandb').setLevel(logging.WARNING)
        
        logger.info("Logging configuration completed")
    
    def setup_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Setup train, validation, and test data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.info("Setting up data loaders...")
        
        # Get data configuration
        data_config = self.config['data']
        training_config = self.config['training']
        
        # Load dataset
        dataset_path = data_config.get('metadata_path')
        if not dataset_path:
            raise ValueError("metadata_path not specified in data configuration")
        
        # Get transforms
        train_transform, val_transform = get_transforms(
            input_size=data_config['input_size'],
            augmentation=not self.debug_mode  # Disable augmentation in debug mode
        )
        
        # Create datasets
        full_dataset = HierarchicalDeepfakeDataset(
            metadata_path=dataset_path,
            transform=None,  # Will apply transforms in data loaders
            validate_files=not self.debug_mode,  # Skip validation in debug mode
            cache_size=data_config.get('cache_size', 1000),
            memory_limit_gb=data_config.get('memory_limit_gb', 4.0)
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(data_config['train_split'] * total_size)
        val_size = int(data_config['val_split'] * total_size)
        test_size = total_size - train_size - val_size
        
        logger.info(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Create splits
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config['reproducibility']['seed'])
        )
        
        # Apply transforms to datasets
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform
        
        # Create data loaders
        common_loader_args = {
            'batch_size': training_config['batch_size'],
            'num_workers': training_config.get('num_workers', 4),
            'pin_memory': training_config.get('pin_memory', True),
            'persistent_workers': training_config.get('num_workers', 4) > 0
        }
        
        # Optimize batch size for distributed training
        if self.distributed_optimizer:
            optimized_batch_size = self.distributed_optimizer.get_optimal_batch_size(
                common_loader_args['batch_size']
            )
            common_loader_args['batch_size'] = optimized_batch_size
            logger.info(f"Optimized batch size for distributed training: {optimized_batch_size}")
        
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **common_loader_args
        )
        
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            **common_loader_args
        )
        
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            drop_last=False,
            **common_loader_args
        )
        
        # Store data loaders
        self.data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        # Log dataset statistics
        class_weights = full_dataset.get_class_weights()
        logger.info(f"Class weights: {class_weights.tolist()}")
        
        # Performance statistics
        if hasattr(full_dataset, 'get_performance_stats'):
            perf_stats = full_dataset.get_performance_stats()
            logger.info(f"Dataset performance: {perf_stats}")
        
        logger.info("Data loaders setup completed")
        return train_loader, val_loader, test_loader
    
    def setup_model(self) -> CRAFTDFModel:
        """
        Setup and initialize the CRAFT-DF model.
        
        Returns:
            Initialized CRAFT-DF model
        """
        logger.info("Setting up CRAFT-DF model...")
        
        # Get model configuration
        model_config = self.config['model']
        training_config = self.config['training']
        
        # Prepare model arguments
        model_args = {
            'spatial_config': {
                'pretrained': model_config.get('spatial_pretrained', True),
                'freeze_layers': model_config.get('spatial_freeze_layers', 10),
                'feature_dim': model_config.get('spatial_feature_dim', 1280),
                'dropout_rate': model_config.get('dropout_rate', 0.1)
            },
            'frequency_config': {
                'input_channels': 3,
                'dwt_levels': model_config.get('freq_dwt_levels', 3),
                'feature_dim': model_config.get('freq_feature_dim', 512),
                'dropout_rate': model_config.get('dropout_rate', 0.1)
            },
            'attention_config': {
                'spatial_dim': model_config.get('spatial_feature_dim', 1280),
                'frequency_dim': model_config.get('freq_feature_dim', 512),
                'embed_dim': model_config.get('attention_dim', 512),
                'num_heads': model_config.get('attention_heads', 8),
                'dropout_rate': model_config.get('dropout_rate', 0.1)
            },
            'num_classes': model_config.get('num_classes', 2),
            'learning_rate': training_config.get('learning_rate', 1e-4),
            'weight_decay': training_config.get('weight_decay', 1e-5),
            'scheduler_type': training_config.get('scheduler_type', 'cosine'),
            'adversarial_training': model_config.get('adversarial_training', True),
            'domain_adaptation_weight': model_config.get('domain_adaptation_weight', 0.1),
            'warmup_epochs': training_config.get('warmup_epochs', 5)
        }
        
        # Add disentanglement configuration if adversarial training is enabled
        if model_args['adversarial_training']:
            model_args['disentanglement_config'] = {
                'input_dim': model_config.get('attention_dim', 512),
                'invariant_dim': model_config.get('invariant_dim', 256),
                'specific_dim': model_config.get('specific_dim', 128),
                'num_domains': model_config.get('num_domains', 4),
                'hidden_dim': model_config.get('disentanglement_hidden_dim', 512),
                'adversarial_weight': model_config.get('domain_adaptation_weight', 0.1),
                'reconstruction_weight': model_config.get('reconstruction_weight', 0.01),
                'gradient_reversal_lambda': model_config.get('gradient_reversal_lambda', 1.0)
            }
        
        # Initialize model
        self.model = CRAFTDFModel(**model_args)
        
        # Apply GPU optimizations
        if torch.cuda.is_available():
            self.model = self.gpu_optimizer.optimize_model(self.model)
            
            # Get mixed precision scaler if enabled
            self.mixed_precision_scaler = self.gpu_optimizer.get_mixed_precision_scaler()
            if self.mixed_precision_scaler:
                logger.info("Mixed precision training enabled with optimized scaler")
        
        # Log model summary
        model_summary = self.model.get_model_summary()
        logger.info(f"Model summary: {model_summary}")
        
        # Load from checkpoint if specified
        if self.resume_from_checkpoint:
            self._load_checkpoint()
        
        logger.info("CRAFT-DF model setup completed")
        return self.model
    
    def _load_checkpoint(self) -> None:
        """Load model from checkpoint."""
        try:
            checkpoint_path = Path(self.resume_from_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                logger.info("Model state loaded from checkpoint")
            else:
                logger.warning("No model state found in checkpoint")
            
            # Log checkpoint info
            if 'epoch' in checkpoint:
                logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'global_step' in checkpoint:
                logger.info(f"Checkpoint global step: {checkpoint['global_step']}")
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint loading failed: {e}")
    
    def setup_trainer(self) -> pl.Trainer:
        """
        Setup PyTorch Lightning trainer with comprehensive configuration.
        
        Returns:
            Configured PyTorch Lightning trainer
        """
        logger.info("Setting up PyTorch Lightning trainer...")
        
        # Get configuration
        training_config = self.config['training']
        logging_config = self.config['logging']
        hardware_config = self.config['hardware']
        
        # Setup logger
        if not self.offline_mode:
            self.logger_instance = WandbLogger(
                project=self.project_name,
                name=self.experiment_name,
                save_dir='./logs',
                log_model=True,
                config=self.config
            )
        else:
            self.logger_instance = None
            logger.info("Running in offline mode - W&B logging disabled")
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Setup profiler
        profiler = self._setup_profiler()
        
        # Setup strategy for distributed training
        strategy = self._setup_strategy(hardware_config)
        
        # Setup distributed optimizer if using multiple GPUs
        if isinstance(strategy, DDPStrategy) or hardware_config.get('devices', 1) > 1:
            world_size = hardware_config.get('devices', 1)
            self.distributed_optimizer = DistributedTrainingOptimizer(
                world_size=world_size,
                rank=0  # Will be set by trainer
            )
            
            # Setup distributed environment
            self.distributed_optimizer.setup_distributed_environment()
        
        # Trainer arguments
        trainer_args = {
            'max_epochs': training_config['max_epochs'],
            'accelerator': hardware_config.get('accelerator', 'gpu'),
            'devices': hardware_config.get('devices', 1),
            'strategy': strategy,
            'precision': training_config.get('precision', 16),
            'gradient_clip_val': training_config.get('gradient_clip_val', 1.0),
            'accumulate_grad_batches': training_config.get('accumulate_grad_batches', 1),
            'log_every_n_steps': logging_config.get('log_every_n_steps', 50),
            'callbacks': callbacks,
            'logger': self.logger_instance,
            'profiler': profiler,
            'enable_checkpointing': True,
            'enable_progress_bar': not self.debug_mode,
            'enable_model_summary': True,
            'deterministic': self.config['reproducibility'].get('deterministic', True),
            'benchmark': self.config['reproducibility'].get('benchmark', False),
            'sync_batchnorm': hardware_config.get('sync_batchnorm', False)
        }
        
        # Debug mode modifications
        if self.debug_mode:
            trainer_args.update({
                'fast_dev_run': False,  # Run full epochs but fewer of them
                'limit_train_batches': 10,  # Limit batches for faster debugging
                'limit_val_batches': 5,
                'num_sanity_val_steps': 2,
                'enable_progress_bar': True  # Show progress in debug mode
            })
        
        # Mixed precision optimization for H100 and other modern GPUs
        if trainer_args['precision'] == 16 and torch.cuda.is_available():
            # Enable additional optimizations for mixed precision
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled mixed precision optimizations for H100")
            
            # Setup gradient compression for distributed training
            if self.distributed_optimizer and self.distributed_optimizer.world_size > 1:
                compression_hook = self.distributed_optimizer.setup_gradient_compression()
                if compression_hook:
                    logger.info("Enabled gradient compression for distributed training")
        
        # Initialize trainer
        self.trainer = pl.Trainer(**trainer_args)
        
        logger.info("PyTorch Lightning trainer setup completed")
        logger.info(f"Trainer configuration: {trainer_args}")
        
        return self.trainer
    
    def _setup_callbacks(self) -> List[pl.Callback]:
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'./checkpoints/{self.experiment_name}',
            filename='{epoch:02d}-{val_accuracy:.3f}',
            monitor=self.config['logging'].get('monitor', 'val_accuracy'),
            mode=self.config['logging'].get('mode', 'max'),
            save_top_k=self.config['logging'].get('save_top_k', 3),
            save_last=True,
            auto_insert_metric_name=False,
            every_n_epochs=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if not self.debug_mode:
            early_stopping = EarlyStopping(
                monitor=self.config['logging'].get('monitor', 'val_accuracy'),
                mode=self.config['logging'].get('mode', 'max'),
                patience=self.config['training'].get('early_stopping_patience', 10),
                min_delta=0.001,
                verbose=True
            )
            callbacks.append(early_stopping)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(
            logging_interval='step',
            log_momentum=True
        )
        callbacks.append(lr_monitor)
        
        # Device stats monitoring (GPU memory, etc.)
        if torch.cuda.is_available():
            device_stats = DeviceStatsMonitor()
            callbacks.append(device_stats)
        
        # Model summary
        model_summary = ModelSummary(max_depth=2)
        callbacks.append(model_summary)
        
        # Performance monitoring callback
        performance_callback = PerformanceCallback(
            log_interval=self.config['logging'].get('log_every_n_steps', 50),
            save_interval=1000,
            enable_profiling=not self.debug_mode,
            profile_steps=100
        )
        callbacks.append(performance_callback)
        
        # Add memory profiling callback for detailed analysis
        if not self.debug_mode and torch.cuda.is_available():
            from .performance_monitor import MemoryProfilingCallback
            memory_callback = MemoryProfilingCallback(
                profile_interval=500,
                save_snapshots=True
            )
            callbacks.append(memory_callback)
        
        logger.info(f"Setup {len(callbacks)} training callbacks")
        return callbacks
    
    def _setup_profiler(self) -> Optional[pl.profilers.Profiler]:
        """Setup performance profiler."""
        profiler_type = self.config.get('profiling', {}).get('type', 'simple')
        
        if profiler_type == 'advanced':
            return AdvancedProfiler(
                dirpath=f'./profiling/{self.experiment_name}',
                filename='profile'
            )
        elif profiler_type == 'simple':
            return SimpleProfiler(
                dirpath=f'./profiling/{self.experiment_name}',
                filename='profile'
            )
        else:
            return None
    
    def _setup_strategy(self, hardware_config: Dict[str, Any]) -> Union[str, pl.strategies.Strategy]:
        """Setup training strategy for distributed training."""
        strategy_name = hardware_config.get('strategy', 'auto')
        
        if strategy_name == 'ddp' and hardware_config.get('devices', 1) > 1:
            # Get optimized settings for distributed training
            if self.distributed_optimizer:
                comm_settings = self.distributed_optimizer.optimize_communication()
                return DDPStrategy(
                    find_unused_parameters=comm_settings.get('find_unused_parameters', False),
                    gradient_as_bucket_view=comm_settings.get('gradient_as_bucket_view', True),
                    bucket_cap_mb=comm_settings.get('bucket_cap_mb', 25)
                )
            else:
                return DDPStrategy(
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True
                )
        else:
            return strategy_name
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info("Starting CRAFT-DF training pipeline...")
        self.training_start_time = time.time()
        
        try:
            # Setup reproducibility
            seed_everything(self.config['reproducibility']['seed'])
            
            # Setup components
            train_loader, val_loader, test_loader = self.setup_data_loaders()
            model = self.setup_model()
            trainer = self.setup_trainer()
            
            # Log system information
            self._log_system_info()
            
            # Start training
            logger.info("Beginning model training...")
            trainer.fit(
                model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=self.resume_from_checkpoint
            )
            
            # Test the model
            if not self.debug_mode:
                logger.info("Running final model evaluation...")
                test_results = trainer.test(
                    model=model,
                    dataloaders=test_loader,
                    ckpt_path='best'
                )
            else:
                test_results = [{'test_accuracy': 0.0, 'test_loss': 0.0}]
            
            # Calculate training time
            training_time = time.time() - self.training_start_time
            
            # Compile results
            results = {
                'experiment_name': self.experiment_name,
                'training_time_seconds': training_time,
                'training_time_formatted': self._format_time(training_time),
                'final_epoch': trainer.current_epoch,
                'global_step': trainer.global_step,
                'test_results': test_results[0] if test_results else {},
                'best_model_path': trainer.checkpoint_callback.best_model_path,
                'config': self.config
            }
            
            # Log final results
            logger.info("Training completed successfully!")
            logger.info(f"Training time: {results['training_time_formatted']}")
            logger.info(f"Final test accuracy: {test_results[0].get('test_accuracy', 'N/A')}")
            logger.info(f"Best model saved at: {results['best_model_path']}")
            
            # Save results
            self._save_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training pipeline failed: {e}")
        
        finally:
            # Cleanup
            if self.logger_instance:
                self.logger_instance.finalize('success')
    
    def _log_system_info(self) -> None:
        """Log comprehensive system information."""
        logger.info("=== System Information ===")
        
        # Python and PyTorch versions
        import sys
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"PyTorch Lightning version: {pl.__version__}")
        
        # Hardware information
        logger.info(f"CPU count: {psutil.cpu_count()}")
        logger.info(f"Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # GPU information
        if torch.cuda.is_available():
            logger.info(f"CUDA available: True")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            
            try:
                gpus = GPUtil.getGPUs() if _GPUTIL_AVAILABLE else []
                for i, gpu in enumerate(gpus):
                    logger.info(f"GPU {i}: {gpu.name} ({gpu.memoryTotal}MB)")
            except:
                logger.info("GPU details unavailable")
        else:
            logger.info("CUDA available: False")
        
        logger.info("=== End System Information ===")
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save training results to file."""
        results_dir = Path(f'./results/{self.experiment_name}')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        results_file = results_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            # Convert non-serializable objects
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, torch.dtype)):
            return str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def evaluate_model(
        self,
        checkpoint_path: str,
        test_data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            checkpoint_path: Path to model checkpoint
            test_data_path: Optional path to test data (uses config if not provided)
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating model from checkpoint: {checkpoint_path}")
        
        # Load model from checkpoint
        model = CRAFTDFModel.load_from_checkpoint(checkpoint_path)
        
        # Setup data loader
        if test_data_path:
            # Use custom test data
            test_dataset = HierarchicalDeepfakeDataset(
                metadata_path=test_data_path,
                transform=get_transforms(self.config['data']['input_size'])[1]  # Use validation transform
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=self.config['training'].get('num_workers', 4)
            )
        else:
            # Use existing test loader
            if 'test' not in self.data_loaders:
                self.setup_data_loaders()
            test_loader = self.data_loaders['test']
        
        # Setup trainer for evaluation
        trainer = pl.Trainer(
            accelerator=self.config['hardware'].get('accelerator', 'gpu'),
            devices=1,  # Single device for evaluation
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True
        )
        
        # Run evaluation
        results = trainer.test(model, test_loader)
        
        logger.info(f"Evaluation completed: {results[0]}")
        return results[0]