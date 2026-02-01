"""
Reproducibility utilities for CRAFT-DF.

This module provides functions to ensure deterministic behavior across
different runs of the model training and inference.
"""

import os
import random
from typing import Optional
import numpy as np
import torch
import pytorch_lightning as pl


def seed_everything(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> int:
    """
    Set seeds for all random number generators to ensure reproducibility.
    
    This function sets seeds for Python's random module, NumPy, PyTorch (CPU and GPU),
    and PyTorch Lightning. It also configures PyTorch's deterministic behavior settings.
    
    Args:
        seed: Random seed value to use across all libraries
        deterministic: If True, enables deterministic algorithms in PyTorch.
                      This may impact performance but ensures reproducibility.
        benchmark: If True, enables cuDNN benchmark mode for potentially faster
                  training but less deterministic behavior.
    
    Returns:
        int: The seed value that was set
        
    Note:
        - Setting deterministic=True may reduce performance but ensures reproducibility
        - Setting benchmark=True may improve performance but reduce reproducibility
        - These settings are mutually exclusive in their effects on reproducibility
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set PyTorch Lightning seed
    pl.seed_everything(seed, workers=True)
    
    # Configure PyTorch deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms (may impact performance)
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark
        torch.use_deterministic_algorithms(False)
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA operations
    
    return seed


def get_reproducibility_info() -> dict:
    """
    Get current reproducibility settings and environment information.
    
    Returns:
        dict: Dictionary containing current reproducibility settings
    """
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'Not set'),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'Not set')
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()
    
    return info


def configure_reproducibility_from_config(config: dict) -> int:
    """
    Configure reproducibility settings from configuration dictionary.
    
    Args:
        config: Configuration dictionary containing reproducibility settings
        
    Returns:
        int: The seed value that was set
        
    Example:
        config = {
            'reproducibility': {
                'seed': 42,
                'deterministic': True,
                'benchmark': False
            }
        }
        seed = configure_reproducibility_from_config(config)
    """
    repro_config = config.get('reproducibility', {})
    
    seed = repro_config.get('seed', 42)
    deterministic = repro_config.get('deterministic', True)
    benchmark = repro_config.get('benchmark', False)
    
    return seed_everything(seed=seed, deterministic=deterministic, benchmark=benchmark)


def verify_reproducibility(model: torch.nn.Module, 
                         input_tensor: torch.Tensor, 
                         num_runs: int = 3) -> bool:
    """
    Verify that model produces consistent outputs across multiple runs.
    
    Args:
        model: PyTorch model to test
        input_tensor: Input tensor for testing
        num_runs: Number of forward passes to compare
        
    Returns:
        bool: True if all outputs are identical, False otherwise
    """
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_tensor)
            outputs.append(output.clone())
    
    # Check if all outputs are identical
    for i in range(1, len(outputs)):
        if not torch.allclose(outputs[0], outputs[i], rtol=1e-6, atol=1e-6):
            return False
    
    return True


class ReproducibilityContext:
    """
    Context manager for temporary reproducibility settings.
    
    Example:
        with ReproducibilityContext(seed=123, deterministic=True):
            # Code that needs specific reproducibility settings
            output = model(input_data)
        # Original settings are restored
    """
    
    def __init__(self, seed: Optional[int] = None, 
                 deterministic: Optional[bool] = None,
                 benchmark: Optional[bool] = None):
        """
        Initialize reproducibility context.
        
        Args:
            seed: Seed to set (if None, current seed is preserved)
            deterministic: Deterministic setting (if None, current setting is preserved)
            benchmark: Benchmark setting (if None, current setting is preserved)
        """
        self.seed = seed
        self.deterministic = deterministic
        self.benchmark = benchmark
        
        # Store original settings
        self.original_deterministic = None
        self.original_benchmark = None
        self.original_rng_state = None
        self.original_cuda_rng_state = None
    
    def __enter__(self):
        """Enter the context and apply new settings."""
        # Store original settings
        self.original_deterministic = torch.backends.cudnn.deterministic
        self.original_benchmark = torch.backends.cudnn.benchmark
        self.original_rng_state = torch.get_rng_state()
        
        if torch.cuda.is_available():
            self.original_cuda_rng_state = torch.cuda.get_rng_state_all()
        
        # Apply new settings
        if self.seed is not None:
            seed_everything(
                seed=self.seed,
                deterministic=self.deterministic if self.deterministic is not None else self.original_deterministic,
                benchmark=self.benchmark if self.benchmark is not None else self.original_benchmark
            )
        else:
            if self.deterministic is not None:
                torch.backends.cudnn.deterministic = self.deterministic
            if self.benchmark is not None:
                torch.backends.cudnn.benchmark = self.benchmark
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore original settings."""
        # Restore original settings
        torch.backends.cudnn.deterministic = self.original_deterministic
        torch.backends.cudnn.benchmark = self.original_benchmark
        torch.set_rng_state(self.original_rng_state)
        
        if torch.cuda.is_available() and self.original_cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(self.original_cuda_rng_state)