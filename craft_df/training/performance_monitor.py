"""
Performance monitoring and GPU optimization utilities for CRAFT-DF training.

This module provides comprehensive performance monitoring, memory profiling,
and GPU optimization features for efficient training on high-performance hardware.
"""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
from collections import deque
import numpy as np

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False
    GPUtil = None

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    epoch: int
    step: int
    
    # Training metrics
    batch_time: float = 0.0
    data_loading_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    optimizer_time: float = 0.0
    
    # Memory metrics (MB)
    gpu_memory_allocated: float = 0.0
    gpu_memory_reserved: float = 0.0
    gpu_memory_free: float = 0.0
    cpu_memory_used: float = 0.0
    cpu_memory_percent: float = 0.0
    
    # GPU utilization
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    gpu_power_draw: float = 0.0
    
    # Throughput metrics
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    
    # Model metrics
    loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0


class GPUOptimizer:
    """
    GPU optimization utilities for CRAFT-DF training.
    
    Provides automatic mixed precision, memory optimization,
    and performance tuning for NVIDIA GPUs including H100 optimizations.
    """
    
    def __init__(
        self,
        enable_amp: bool = True,
        memory_fraction: float = 0.9,
        allow_growth: bool = True,
        optimize_for_inference: bool = False,
        enable_h100_optimizations: bool = True
    ):
        self.enable_amp = enable_amp
        self.memory_fraction = memory_fraction
        self.allow_growth = allow_growth
        self.optimize_for_inference = optimize_for_inference
        self.enable_h100_optimizations = enable_h100_optimizations
        
        # H100 specific settings
        self.h100_detected = self._detect_h100()
        
        self._setup_gpu_optimization()
    
    def _detect_h100(self) -> bool:
        """Detect if H100 GPU is available."""
        if not torch.cuda.is_available():
            return False
        
        try:
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i).lower()
                if 'h100' in gpu_name:
                    logger.info(f"H100 GPU detected: {torch.cuda.get_device_name(i)}")
                    return True
        except Exception as e:
            logger.debug(f"GPU detection error: {e}")
        
        return False
    
    def _setup_gpu_optimization(self) -> None:
        """Setup GPU optimization settings."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - GPU optimization disabled")
            return
        
        # Set memory fraction
        if self.memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            logger.info(f"Set GPU memory fraction to {self.memory_fraction}")
        
        # Enable memory growth
        if self.allow_growth:
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled CUDNN benchmark for dynamic input sizes")
        
        # Optimize for inference if specified
        if self.optimize_for_inference:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            logger.info("Optimized CUDNN settings for inference")
        
        # Enable Tensor Core usage
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 for Tensor Core acceleration")
        
        # H100 specific optimizations
        if self.h100_detected and self.enable_h100_optimizations:
            self._setup_h100_optimizations()
    
    def _setup_h100_optimizations(self) -> None:
        """Setup H100-specific optimizations."""
        try:
            # Enable Flash Attention if available
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("Enabled Flash Attention for H100")
            
            # Optimize for FP8 if supported
            if hasattr(torch.backends.cuda, 'enable_fp8'):
                torch.backends.cuda.enable_fp8 = True
                logger.info("Enabled FP8 optimization for H100")
            
            # Set optimal memory pool settings for H100
            if hasattr(torch.cuda, 'set_memory_pool_settings'):
                torch.cuda.set_memory_pool_settings(
                    max_split_size_mb=512,  # Optimize for H100's memory bandwidth
                    roundup_power2_divisions=16
                )
                logger.info("Optimized memory pool settings for H100")
            
            # Enable compilation optimizations for H100
            if hasattr(torch, '_dynamo'):
                torch._dynamo.config.cache_size_limit = 256
                torch._dynamo.config.optimize_ddp = True
                logger.info("Enabled dynamo optimizations for H100")
                
        except Exception as e:
            logger.warning(f"H100 optimization setup failed: {e}")
    
    def get_mixed_precision_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """Get optimized GradScaler for mixed precision training."""
        if not self.enable_amp or not torch.cuda.is_available():
            return None
        
        # H100 optimized scaler settings
        if self.h100_detected:
            return torch.cuda.amp.GradScaler(
                init_scale=2.**16,  # Higher initial scale for H100
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=1000,  # More frequent scaling updates
                enabled=True
            )
        else:
            return torch.cuda.amp.GradScaler(enabled=True)
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply model-level optimizations.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        # Compile model for PyTorch 2.0+
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
            try:
                # H100 optimized compilation
                if self.h100_detected:
                    model = torch.compile(
                        model, 
                        mode='max-autotune-no-cudagraphs',  # Best for H100
                        fullgraph=True,
                        dynamic=False
                    )
                    logger.info("Applied H100-optimized torch.compile")
                else:
                    model = torch.compile(model, mode='max-autotune')
                    logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"torch.compile failed, continuing without compilation: {e}")
                # Continue without compilation - model is still functional
        
        # Enable channels_last memory format for better performance
        if torch.cuda.is_available():
            try:
                model = model.to(memory_format=torch.channels_last)
                logger.info("Applied channels_last memory format")
            except Exception as e:
                logger.warning(f"channels_last optimization failed: {e}")
        
        # Apply FSDP for large models if beneficial
        if self._should_use_fsdp(model):
            model = self._apply_fsdp(model)
        
        return model
    
    def _should_use_fsdp(self, model: nn.Module) -> bool:
        """Determine if FSDP should be used for the model."""
        try:
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            
            # Use FSDP for models > 100M parameters or if multiple GPUs available
            return param_count > 100_000_000 or torch.cuda.device_count() > 1
        except Exception:
            return False
    
    def _apply_fsdp(self, model: nn.Module) -> nn.Module:
        """Apply Fully Sharded Data Parallel to model."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            
            # Configure FSDP with H100 optimizations
            fsdp_config = {
                'auto_wrap_policy': transformer_auto_wrap_policy,
                'mixed_precision': self._get_fsdp_mixed_precision(),
                'backward_prefetch': FSDP.BackwardPrefetch.BACKWARD_PRE,
                'forward_prefetch': True,
                'limit_all_gathers': True,
                'use_orig_params': True
            }
            
            model = FSDP(model, **fsdp_config)
            logger.info("Applied FSDP optimization")
            return model
            
        except ImportError:
            logger.warning("FSDP not available, skipping")
            return model
        except Exception as e:
            logger.warning(f"FSDP application failed: {e}")
            return model
    
    def _get_fsdp_mixed_precision(self):
        """Get FSDP mixed precision configuration."""
        try:
            from torch.distributed.fsdp import MixedPrecision
            
            if self.enable_amp:
                return MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )
            else:
                return None
        except ImportError:
            return None
    
    def get_optimal_batch_size(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        max_memory_gb: float = 8.0,
        start_batch_size: int = 1
    ) -> int:
        """
        Find optimal batch size for given model and input shape.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (without batch dimension)
            max_memory_gb: Maximum GPU memory to use (GB)
            start_batch_size: Starting batch size for search
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return start_batch_size
        
        model.eval()
        device = next(model.parameters()).device
        
        batch_size = start_batch_size
        max_batch_size = start_batch_size
        
        while batch_size <= 1024:  # Reasonable upper limit
            try:
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape, device=device)
                
                # Forward pass
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated(device) / (1024**3)  # GB
                
                if memory_used > max_memory_gb:
                    break
                
                max_batch_size = batch_size
                batch_size *= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
            finally:
                torch.cuda.empty_cache()
        
        logger.info(f"Optimal batch size found: {max_batch_size}")
        return max_batch_size


class MemoryProfiler:
    """
    Advanced memory profiling for GPU and CPU usage.
    
    Provides detailed memory analysis including peak usage,
    fragmentation analysis, and memory leak detection.
    """
    
    def __init__(self, enable_detailed_profiling: bool = True):
        self.enable_detailed_profiling = enable_detailed_profiling
        self.memory_snapshots = []
        self.peak_memory = {'gpu': 0, 'cpu': 0}
        
        # Initialize NVML if available
        self.nvml_available = NVML_AVAILABLE
        if self.nvml_available:
            try:
                pynvml.nvmlInit()
                self.gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) 
                    for i in range(torch.cuda.device_count())
                ]
            except Exception as e:
                logger.warning(f"NVML initialization failed: {e}")
                self.nvml_available = False
                self.gpu_handles = []
    
    def take_snapshot(self, tag: str = "") -> Dict[str, Any]:
        """Take a detailed memory snapshot."""
        snapshot = {
            'timestamp': time.time(),
            'tag': tag,
            'cpu_memory': self._get_cpu_memory_info(),
            'gpu_memory': self._get_gpu_memory_info() if torch.cuda.is_available() else {}
        }
        
        self.memory_snapshots.append(snapshot)
        
        # Update peak memory tracking
        if snapshot['cpu_memory']['used_mb'] > self.peak_memory['cpu']:
            self.peak_memory['cpu'] = snapshot['cpu_memory']['used_mb']
        
        if torch.cuda.is_available() and 'allocated_mb' in snapshot['gpu_memory']:
            if snapshot['gpu_memory']['allocated_mb'] > self.peak_memory['gpu']:
                self.peak_memory['gpu'] = snapshot['gpu_memory']['allocated_mb']
        
        return snapshot
    
    def _get_cpu_memory_info(self) -> Dict[str, float]:
        """Get detailed CPU memory information."""
        memory_info = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'total_mb': memory_info.total / (1024**2),
            'available_mb': memory_info.available / (1024**2),
            'used_mb': memory_info.used / (1024**2),
            'percent': memory_info.percent,
            'process_rss_mb': process_memory.rss / (1024**2),
            'process_vms_mb': process_memory.vms / (1024**2)
        }
    
    def _get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get detailed GPU memory information."""
        if not torch.cuda.is_available():
            return {}
        
        device = torch.cuda.current_device()
        
        # PyTorch memory info
        memory_info = {
            'allocated_mb': torch.cuda.memory_allocated(device) / (1024**2),
            'reserved_mb': torch.cuda.memory_reserved(device) / (1024**2),
            'max_allocated_mb': torch.cuda.max_memory_allocated(device) / (1024**2),
            'max_reserved_mb': torch.cuda.max_memory_reserved(device) / (1024**2)
        }
        
        # NVML memory info if available
        if self.nvml_available and device < len(self.gpu_handles):
            try:
                handle = self.gpu_handles[device]
                nvml_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                memory_info.update({
                    'total_mb': nvml_memory.total / (1024**2),
                    'free_mb': nvml_memory.free / (1024**2),
                    'used_mb': nvml_memory.used / (1024**2),
                    'utilization_percent': pynvml.nvmlDeviceGetUtilizationRates(handle).memory
                })
            except Exception as e:
                logger.debug(f"NVML memory query failed: {e}")
        
        # Memory fragmentation analysis
        if self.enable_detailed_profiling:
            memory_info['fragmentation_ratio'] = self._calculate_fragmentation_ratio()
        
        return memory_info
    
    def _calculate_fragmentation_ratio(self) -> float:
        """Calculate GPU memory fragmentation ratio."""
        try:
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            
            if reserved > 0:
                return 1.0 - (allocated / reserved)
            return 0.0
        except Exception:
            return 0.0
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns from snapshots."""
        if not self.memory_snapshots:
            return {'error': 'No memory snapshots available'}
        
        # Extract time series data
        timestamps = [s['timestamp'] for s in self.memory_snapshots]
        cpu_usage = [s['cpu_memory']['used_mb'] for s in self.memory_snapshots]
        
        analysis = {
            'duration_seconds': timestamps[-1] - timestamps[0],
            'num_snapshots': len(self.memory_snapshots),
            'peak_memory': self.peak_memory.copy(),
            'cpu_memory': {
                'initial_mb': cpu_usage[0],
                'final_mb': cpu_usage[-1],
                'peak_mb': max(cpu_usage),
                'growth_mb': cpu_usage[-1] - cpu_usage[0],
                'average_mb': sum(cpu_usage) / len(cpu_usage)
            }
        }
        
        # GPU analysis if available
        if torch.cuda.is_available() and 'gpu_memory' in self.memory_snapshots[0]:
            gpu_allocated = [
                s['gpu_memory'].get('allocated_mb', 0) 
                for s in self.memory_snapshots
            ]
            
            if gpu_allocated and any(gpu_allocated):
                analysis['gpu_memory'] = {
                    'initial_mb': gpu_allocated[0],
                    'final_mb': gpu_allocated[-1],
                    'peak_mb': max(gpu_allocated),
                    'growth_mb': gpu_allocated[-1] - gpu_allocated[0],
                    'average_mb': sum(gpu_allocated) / len(gpu_allocated)
                }
                
                # Fragmentation analysis
                fragmentation_ratios = [
                    s['gpu_memory'].get('fragmentation_ratio', 0)
                    for s in self.memory_snapshots
                ]
                
                if any(fragmentation_ratios):
                    analysis['gpu_memory']['fragmentation'] = {
                        'average_ratio': sum(fragmentation_ratios) / len(fragmentation_ratios),
                        'max_ratio': max(fragmentation_ratios),
                        'final_ratio': fragmentation_ratios[-1]
                    }
        
        # Memory leak detection
        analysis['leak_detection'] = self._detect_memory_leaks()
        
        return analysis
    
    def _detect_memory_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks from usage patterns."""
        if len(self.memory_snapshots) < 10:
            return {'status': 'insufficient_data'}
        
        # Analyze last 50% of snapshots for trend
        mid_point = len(self.memory_snapshots) // 2
        recent_snapshots = self.memory_snapshots[mid_point:]
        
        cpu_usage = [s['cpu_memory']['used_mb'] for s in recent_snapshots]
        
        # Calculate trend
        x = np.arange(len(cpu_usage))
        cpu_trend = np.polyfit(x, cpu_usage, 1)[0]  # Linear trend coefficient
        
        leak_detection = {
            'cpu_trend_mb_per_snapshot': float(cpu_trend),
            'cpu_leak_suspected': cpu_trend > 5.0,  # >5MB growth per snapshot
        }
        
        # GPU leak detection
        if torch.cuda.is_available():
            gpu_usage = [
                s['gpu_memory'].get('allocated_mb', 0) 
                for s in recent_snapshots
            ]
            
            if any(gpu_usage):
                gpu_trend = np.polyfit(x, gpu_usage, 1)[0]
                leak_detection.update({
                    'gpu_trend_mb_per_snapshot': float(gpu_trend),
                    'gpu_leak_suspected': gpu_trend > 10.0  # >10MB growth per snapshot
                })
        
        return leak_detection
    
    def save_profile(self, filepath: str) -> None:
        """Save memory profile to file."""
        profile_data = {
            'analysis': self.analyze_memory_usage(),
            'snapshots': self.memory_snapshots,
            'metadata': {
                'profiler_version': '1.0',
                'timestamp': time.time(),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
        
        logger.info(f"Memory profile saved to {filepath}")


class PerformanceMonitor:
    """
    Comprehensive performance monitoring for training.
    
    Tracks GPU utilization, memory usage, throughput metrics,
    and provides real-time performance analysis.
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        save_interval: int = 1000,
        max_history: int = 10000,
        monitor_gpu: bool = True,
        monitor_cpu: bool = True,
        profile_memory: bool = True,
        enable_detailed_profiling: bool = False
    ):
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.max_history = max_history
        self.monitor_gpu = monitor_gpu and torch.cuda.is_available()
        self.monitor_cpu = monitor_cpu
        self.profile_memory = profile_memory
        self.enable_detailed_profiling = enable_detailed_profiling
        
        # Performance history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.step_counter = 0
        
        # Timing contexts
        self.timers = {}
        
        # Memory profiler
        self.memory_profiler = MemoryProfiler(enable_detailed_profiling) if profile_memory else None
        
        # GPU monitoring setup
        if self.monitor_gpu and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handles = []
                for i in range(torch.cuda.device_count()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
                logger.info(f"Initialized NVML monitoring for {len(self.gpu_handles)} GPUs")
            except Exception as e:
                logger.warning(f"NVML initialization failed: {e}")
                self.gpu_handles = []
        else:
            self.gpu_handles = []
        
        # Background monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
        # Performance benchmarking
        self.benchmark_results = {}
        
        logger.info("PerformanceMonitor initialized")
    
    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self._monitor_thread.start()
        logger.info("Started background performance monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped background performance monitoring")
    
    def _background_monitor(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                # Collect system metrics
                if self.monitor_cpu:
                    cpu_percent = psutil.cpu_percent(interval=1.0)
                    memory_info = psutil.virtual_memory()
                
                if self.monitor_gpu and self.gpu_handles:
                    for i, handle in enumerate(self.gpu_handles):
                        try:
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                            
                            # Log GPU metrics periodically
                            if self.step_counter % (self.log_interval * 10) == 0:
                                logger.debug(f"GPU {i}: {util.gpu}% util, {temp}°C, {power:.1f}W")
                                
                        except Exception as e:
                            logger.debug(f"GPU monitoring error: {e}")
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                break
    
    def record_step_metrics(
        self,
        epoch: int,
        step: int,
        batch_size: int,
        loss: float = 0.0,
        learning_rate: float = 0.0,
        **kwargs
    ) -> PerformanceMetrics:
        """
        Record metrics for a training step.
        
        Args:
            epoch: Current epoch
            step: Current step
            batch_size: Batch size
            loss: Training loss
            learning_rate: Current learning rate
            **kwargs: Additional metrics
            
        Returns:
            PerformanceMetrics object
        """
        timestamp = time.time()
        
        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=learning_rate
        )
        
        # GPU memory metrics
        if self.monitor_gpu and torch.cuda.is_available():
            device = torch.cuda.current_device()
            metrics.gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
            metrics.gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)  # MB
            
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # MB
            metrics.gpu_memory_free = total_memory - metrics.gpu_memory_reserved
        
        # CPU memory metrics
        if self.monitor_cpu:
            memory_info = psutil.virtual_memory()
            metrics.cpu_memory_used = memory_info.used / (1024**2)  # MB
            metrics.cpu_memory_percent = memory_info.percent
        
        # GPU utilization (if available)
        if self.gpu_handles and len(self.gpu_handles) > 0:
            try:
                handle = self.gpu_handles[0]  # Use first GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.gpu_utilization = util.gpu
                metrics.gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics.gpu_power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
            except Exception as e:
                logger.debug(f"GPU utilization monitoring error: {e}")
        
        # Throughput metrics
        if hasattr(self, '_last_step_time') and self._last_step_time:
            step_time = timestamp - self._last_step_time
            metrics.samples_per_second = batch_size / step_time if step_time > 0 else 0.0
        
        self._last_step_time = timestamp
        
        # Add custom metrics
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        
        # Store in history
        self.metrics_history.append(metrics)
        self.step_counter += 1
        
        # Log periodically
        if self.step_counter % self.log_interval == 0:
            self._log_metrics(metrics)
        
        # Save periodically
        if self.step_counter % self.save_interval == 0:
            self.save_metrics()
        
        return metrics
    
    def _log_metrics(self, metrics: PerformanceMetrics) -> None:
        """Log performance metrics."""
        logger.info(
            f"Step {metrics.step}: "
            f"Loss={metrics.loss:.4f}, "
            f"GPU Mem={metrics.gpu_memory_allocated:.0f}MB, "
            f"GPU Util={metrics.gpu_utilization:.1f}%, "
            f"Samples/s={metrics.samples_per_second:.1f}"
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary containing performance statistics
        """
        if not self.metrics_history:
            return {"message": "No performance data available"}
        
        # Convert to numpy arrays for analysis
        gpu_memory = np.array([m.gpu_memory_allocated for m in self.metrics_history])
        gpu_util = np.array([m.gpu_utilization for m in self.metrics_history])
        samples_per_sec = np.array([m.samples_per_second for m in self.metrics_history if m.samples_per_second > 0])
        
        summary = {
            "total_steps": len(self.metrics_history),
            "monitoring_duration_hours": (self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp) / 3600,
            
            "gpu_memory_mb": {
                "mean": float(np.mean(gpu_memory)),
                "max": float(np.max(gpu_memory)),
                "min": float(np.min(gpu_memory)),
                "std": float(np.std(gpu_memory))
            },
            
            "gpu_utilization_percent": {
                "mean": float(np.mean(gpu_util)),
                "max": float(np.max(gpu_util)),
                "min": float(np.min(gpu_util)),
                "std": float(np.std(gpu_util))
            },
            
            "throughput": {
                "mean_samples_per_second": float(np.mean(samples_per_sec)) if len(samples_per_sec) > 0 else 0.0,
                "max_samples_per_second": float(np.max(samples_per_sec)) if len(samples_per_sec) > 0 else 0.0,
                "total_samples_processed": int(np.sum(samples_per_sec)) if len(samples_per_sec) > 0 else 0
            }
        }
        
        return summary
    
    def save_metrics(self, filepath: Optional[str] = None) -> None:
        """Save performance metrics to file."""
        if not self.metrics_history:
            return
        
        if filepath is None:
            filepath = f"performance_metrics_{int(time.time())}.json"
        
        # Convert metrics to serializable format
        metrics_data = []
        for metrics in self.metrics_history:
            metrics_dict = {
                "timestamp": metrics.timestamp,
                "epoch": metrics.epoch,
                "step": metrics.step,
                "gpu_memory_allocated": metrics.gpu_memory_allocated,
                "gpu_utilization": metrics.gpu_utilization,
                "samples_per_second": metrics.samples_per_second,
                "loss": metrics.loss,
                "learning_rate": metrics.learning_rate
            }
            metrics_data.append(metrics_dict)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump({
                "summary": self.get_performance_summary(),
                "metrics": metrics_data
            }, f, indent=2)
        
        logger.info(f"Saved performance metrics to {filepath}")
    
    def run_throughput_benchmark(
        self, 
        model: nn.Module, 
        sample_input: Dict[str, torch.Tensor],
        num_warmup: int = 10,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Run comprehensive throughput benchmark.
        
        Args:
            model: Model to benchmark
            sample_input: Sample input batch
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Running throughput benchmark...")
        
        model.eval()
        device = next(model.parameters()).device
        
        # Move inputs to device
        for key in sample_input:
            if isinstance(sample_input[key], torch.Tensor):
                sample_input[key] = sample_input[key].to(device)
            elif isinstance(sample_input[key], dict):
                for subkey in sample_input[key]:
                    sample_input[key][subkey] = sample_input[key][subkey].to(device)
        
        batch_size = sample_input['spatial_input'].shape[0] if 'spatial_input' in sample_input else 1
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(num_warmup):
                if isinstance(sample_input, dict) and 'frequency_input' in sample_input:
                    _ = model(
                        sample_input['spatial_input'],
                        sample_input['frequency_input'],
                        sample_input.get('domain_labels')
                    )
                else:
                    _ = model(sample_input)
        
        # Benchmark runs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                if isinstance(sample_input, dict) and 'frequency_input' in sample_input:
                    _ = model(
                        sample_input['spatial_input'],
                        sample_input['frequency_input'],
                        sample_input.get('domain_labels')
                    )
                else:
                    _ = model(sample_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        total_samples = num_runs * batch_size
        
        results = {
            'total_time_seconds': total_time,
            'total_samples': total_samples,
            'samples_per_second': total_samples / total_time,
            'batch_time_ms': (total_time / num_runs) * 1000,
            'latency_per_sample_ms': (total_time / total_samples) * 1000
        }
        
        self.benchmark_results['throughput'] = results
        logger.info(f"Throughput benchmark completed: {results['samples_per_second']:.1f} samples/s")
        
        return results
    
    def run_memory_benchmark(
        self,
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        num_steps: int = 20
    ) -> Dict[str, Any]:
        """
        Run memory usage benchmark with detailed profiling.
        
        Args:
            model: Model to benchmark
            sample_input: Sample input batch
            num_steps: Number of training steps to simulate
            
        Returns:
            Dictionary with memory benchmark results
        """
        logger.info("Running memory benchmark...")
        
        if self.memory_profiler is None:
            self.memory_profiler = MemoryProfiler(enable_detailed_profiling=True)
        
        device = next(model.parameters()).device
        
        # Move inputs to device
        for key in sample_input:
            if isinstance(sample_input[key], torch.Tensor):
                sample_input[key] = sample_input[key].to(device)
            elif isinstance(sample_input[key], dict):
                for subkey in sample_input[key]:
                    sample_input[key][subkey] = sample_input[key][subkey].to(device)
        
        # Clear memory and take initial snapshot
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.memory_profiler.take_snapshot("initial")
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Simulate training steps
        for step in range(num_steps):
            self.memory_profiler.take_snapshot(f"step_{step}_start")
            
            # Forward pass
            if isinstance(sample_input, dict) and 'frequency_input' in sample_input:
                outputs = model(
                    sample_input['spatial_input'],
                    sample_input['frequency_input'],
                    sample_input.get('domain_labels')
                )
            else:
                outputs = model(sample_input)
            
            self.memory_profiler.take_snapshot(f"step_{step}_forward")
            
            # Compute loss
            if isinstance(outputs, dict):
                loss = outputs['logits'].sum()  # Dummy loss
            else:
                loss = outputs.sum()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            self.memory_profiler.take_snapshot(f"step_{step}_backward")
            
            optimizer.step()
            
            self.memory_profiler.take_snapshot(f"step_{step}_end")
        
        # Final snapshot
        self.memory_profiler.take_snapshot("final")
        
        # Analyze results
        analysis = self.memory_profiler.analyze_memory_usage()
        
        self.benchmark_results['memory'] = analysis
        logger.info(f"Memory benchmark completed. Peak GPU: {analysis['peak_memory']['gpu']:.1f}MB")
        
        return analysis
    
    def run_distributed_benchmark(
        self,
        world_size: int,
        model_config: Dict[str, Any],
        batch_sizes: List[int] = [8, 16, 32, 64]
    ) -> Dict[str, Any]:
        """
        Benchmark distributed training performance.
        
        Args:
            world_size: Number of processes/GPUs
            model_config: Model configuration
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with distributed benchmark results
        """
        logger.info(f"Running distributed benchmark for world_size={world_size}")
        
        results = {
            'world_size': world_size,
            'batch_size_results': {},
            'optimal_batch_size': None,
            'scaling_efficiency': None
        }
        
        # Test different batch sizes
        best_throughput = 0
        optimal_batch_size = batch_sizes[0]
        
        for batch_size in batch_sizes:
            try:
                # Simulate distributed training performance
                distributed_optimizer = DistributedTrainingOptimizer(world_size=world_size, rank=0)
                
                # Get optimized batch size
                optimized_batch_size = distributed_optimizer.get_optimal_batch_size(batch_size)
                
                # Estimate throughput (simplified calculation)
                # In real scenario, this would involve actual distributed training
                base_throughput = 100  # samples/second baseline
                
                # Account for communication overhead
                communication_overhead = 1.0 - (0.1 * (world_size - 1))  # 10% overhead per additional GPU
                estimated_throughput = base_throughput * world_size * communication_overhead
                
                batch_result = {
                    'original_batch_size': batch_size,
                    'optimized_batch_size': optimized_batch_size,
                    'estimated_throughput': estimated_throughput,
                    'communication_overhead': 1.0 - communication_overhead
                }
                
                results['batch_size_results'][batch_size] = batch_result
                
                if estimated_throughput > best_throughput:
                    best_throughput = estimated_throughput
                    optimal_batch_size = batch_size
                    
            except Exception as e:
                logger.warning(f"Distributed benchmark failed for batch_size={batch_size}: {e}")
        
        results['optimal_batch_size'] = optimal_batch_size
        
        # Calculate scaling efficiency (theoretical vs actual speedup)
        if world_size > 1:
            single_gpu_throughput = results['batch_size_results'][optimal_batch_size]['estimated_throughput'] / world_size
            actual_speedup = best_throughput / single_gpu_throughput
            theoretical_speedup = world_size
            results['scaling_efficiency'] = actual_speedup / theoretical_speedup
        
        self.benchmark_results['distributed'] = results
        logger.info(f"Distributed benchmark completed. Optimal batch size: {optimal_batch_size}")
        
        return results


class PerformanceCallback(Callback):
    """
    PyTorch Lightning callback for performance monitoring.
    
    Integrates PerformanceMonitor with PyTorch Lightning training loop
    to provide automatic performance tracking and optimization.
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        save_interval: int = 1000,
        enable_profiling: bool = False,
        profile_steps: int = 100
    ):
        super().__init__()
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.enable_profiling = enable_profiling
        self.profile_steps = profile_steps
        
        self.monitor = PerformanceMonitor(
            log_interval=log_interval,
            save_interval=save_interval
        )
        
        self.profiler = None
        self.step_times = deque(maxlen=100)
        
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training starts."""
        self.monitor.start_monitoring()
        
        if self.enable_profiling:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.profiler.start()
        
        logger.info("Performance monitoring started")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training ends."""
        self.monitor.stop_monitoring()
        
        if self.profiler:
            self.profiler.stop()
            
            # Save profiler results
            profile_path = f"torch_profiler_{int(time.time())}.json"
            self.profiler.export_chrome_trace(profile_path)
            logger.info(f"Saved torch profiler trace to {profile_path}")
        
        # Save final performance summary
        summary = self.monitor.get_performance_summary()
        logger.info(f"Training performance summary: {summary}")
        
        self.monitor.save_metrics("final_performance_metrics.json")
    
    def on_train_batch_start(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Called at the start of each training batch."""
        self.batch_start_time = time.time()
    
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Called at the end of each training batch."""
        batch_time = time.time() - self.batch_start_time
        self.step_times.append(batch_time)
        
        # Get batch size
        if isinstance(batch, dict):
            batch_size = batch['spatial_input'].shape[0]
        elif isinstance(batch, (list, tuple)):
            batch_size = batch[0].shape[0]
        else:
            batch_size = 1
        
        # Record metrics
        loss = outputs.get('loss', 0.0) if isinstance(outputs, dict) else 0.0
        
        metrics = self.monitor.record_step_metrics(
            epoch=trainer.current_epoch,
            step=trainer.global_step,
            batch_size=batch_size,
            loss=float(loss) if torch.is_tensor(loss) else loss,
            learning_rate=trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else 0.0,
            batch_time=batch_time
        )
        
        # Log to trainer
        if trainer.global_step % self.log_interval == 0:
            pl_module.log_dict({
                'perf/gpu_memory_mb': metrics.gpu_memory_allocated,
                'perf/gpu_utilization': metrics.gpu_utilization,
                'perf/samples_per_second': metrics.samples_per_second,
                'perf/batch_time': batch_time
            }, on_step=True, on_epoch=False)


class DistributedTrainingOptimizer:
    """
    Optimization utilities for distributed training.
    
    Provides configuration and optimization for multi-GPU training
    with proper synchronization and communication optimization.
    """
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.is_distributed = world_size > 1
        
        # Communication optimization settings
        self.communication_backend = self._detect_optimal_backend()
        
    def _detect_optimal_backend(self) -> str:
        """Detect optimal communication backend."""
        if torch.cuda.is_available():
            return 'nccl'  # Best for GPU communication
        else:
            return 'gloo'  # Fallback for CPU
    
    def setup_distributed_environment(self) -> None:
        """Setup distributed training environment."""
        if not self.is_distributed:
            return
        
        try:
            import torch.distributed as dist
            
            # Initialize process group if not already initialized
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.communication_backend,
                    world_size=self.world_size,
                    rank=self.rank
                )
                logger.info(f"Initialized distributed training: rank={self.rank}, world_size={self.world_size}")
            
            # Set device for current process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.rank % torch.cuda.device_count())
                
        except Exception as e:
            logger.error(f"Failed to setup distributed environment: {e}")
            raise
    
    def optimize_communication(self) -> Dict[str, Any]:
        """
        Get optimized settings for distributed communication.
        
        Returns:
            Dictionary of optimized settings
        """
        settings = {
            'find_unused_parameters': False,  # Faster if all parameters are used
            'gradient_as_bucket_view': True,  # Memory optimization
            'static_graph': True,  # If model structure doesn't change
        }
        
        # Optimize based on world size
        if self.world_size <= 4:
            settings['bucket_cap_mb'] = 25  # Default
        elif self.world_size <= 8:
            settings['bucket_cap_mb'] = 50  # Larger buckets for more GPUs
        else:
            settings['bucket_cap_mb'] = 100  # Even larger for many GPUs
        
        # Advanced optimizations for large scale
        if self.world_size >= 16:
            settings.update({
                'ddp_comm_hook': 'fp16_compress',  # Compress gradients
                'broadcast_buffers': False,  # Reduce communication
                'sync_module_states': True,  # Ensure consistency
            })
        
        return settings
    
    def get_optimal_batch_size(self, base_batch_size: int) -> int:
        """
        Calculate optimal batch size for distributed training.
        
        Args:
            base_batch_size: Base batch size for single GPU
            
        Returns:
            Optimal batch size per GPU
        """
        if not self.is_distributed:
            return base_batch_size
        
        # Linear scaling rule with adjustments for communication overhead
        if self.world_size <= 4:
            return base_batch_size
        elif self.world_size <= 8:
            return max(base_batch_size // 2, 8)  # Reduce batch size for many GPUs
        else:
            return max(base_batch_size // 4, 4)  # Further reduction for very large setups
    
    def get_learning_rate_scaling(self, base_lr: float) -> float:
        """
        Calculate scaled learning rate for distributed training.
        
        Args:
            base_lr: Base learning rate for single GPU
            
        Returns:
            Scaled learning rate
        """
        if not self.is_distributed:
            return base_lr
        
        # Linear scaling with square root adjustment for large scale
        if self.world_size <= 8:
            return base_lr * self.world_size
        else:
            # Square root scaling for very large scale to maintain stability
            return base_lr * np.sqrt(self.world_size)
    
    def setup_gradient_compression(self) -> Optional[Any]:
        """Setup gradient compression for communication efficiency."""
        if not self.is_distributed or self.world_size < 8:
            return None
        
        try:
            # Try to setup FP16 gradient compression
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            
            if hasattr(default_hooks, 'fp16_compress_hook'):
                return default_hooks.fp16_compress_hook
            else:
                logger.warning("FP16 compression not available")
                return None
                
        except ImportError:
            logger.warning("Gradient compression hooks not available")
            return None
    
    def get_data_loader_settings(self) -> Dict[str, Any]:
        """Get optimized data loader settings for distributed training."""
        settings = {
            'shuffle': False,  # DistributedSampler handles shuffling
            'drop_last': True,  # Ensure consistent batch sizes across ranks
            'pin_memory': True,  # Faster GPU transfer
        }
        
        # Optimize num_workers based on world size
        if self.world_size <= 4:
            settings['num_workers'] = 4
        elif self.world_size <= 8:
            settings['num_workers'] = 2  # Reduce to avoid CPU bottleneck
        else:
            settings['num_workers'] = 1  # Minimal for very large scale
        
        return settings
    
    def create_distributed_sampler(self, dataset, shuffle: bool = True):
        """Create distributed sampler for dataset."""
        if not self.is_distributed:
            return None
        
        try:
            from torch.utils.data.distributed import DistributedSampler
            
            return DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=True
            )
        except Exception as e:
            logger.error(f"Failed to create distributed sampler: {e}")
            return None
    
    def synchronize_metrics(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Synchronize metrics across all processes."""
        if not self.is_distributed:
            return metrics
        
        try:
            import torch.distributed as dist
            
            synchronized_metrics = {}
            
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    # Ensure tensor is on correct device
                    if torch.cuda.is_available():
                        value = value.cuda()
                    
                    # All-reduce to get average across all processes
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
                    synchronized_metrics[key] = value / self.world_size
                else:
                    synchronized_metrics[key] = value
            
            return synchronized_metrics
            
        except Exception as e:
            logger.warning(f"Metric synchronization failed: {e}")
            return metrics
    
    def cleanup(self) -> None:
        """Cleanup distributed training resources."""
        if not self.is_distributed:
            return
        
        try:
            import torch.distributed as dist
            
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info("Cleaned up distributed training resources")
                
        except Exception as e:
            logger.warning(f"Distributed cleanup failed: {e}")


class MemoryProfilingCallback(Callback):
    """
    PyTorch Lightning callback for detailed memory profiling.
    
    Provides comprehensive memory analysis during training including
    peak usage tracking, fragmentation analysis, and leak detection.
    """
    
    def __init__(
        self,
        profile_interval: int = 500,
        save_snapshots: bool = True,
        enable_leak_detection: bool = True
    ):
        super().__init__()
        self.profile_interval = profile_interval
        self.save_snapshots = save_snapshots
        self.enable_leak_detection = enable_leak_detection
        
        self.memory_profiler = None
        self.step_counter = 0
        
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize memory profiler when training starts."""
        self.memory_profiler = MemoryProfiler(enable_detailed_profiling=True)
        self.memory_profiler.take_snapshot("training_start")
        logger.info("Memory profiling started")
    
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Take memory snapshots at regular intervals."""
        self.step_counter += 1
        
        if self.step_counter % self.profile_interval == 0:
            if self.memory_profiler:
                self.memory_profiler.take_snapshot(f"step_{trainer.global_step}")
                
                # Log current memory usage
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                    
                    pl_module.log_dict({
                        'memory/current_gpu_mb': current_memory,
                        'memory/peak_gpu_mb': peak_memory,
                    }, on_step=True, on_epoch=False)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Analyze memory usage at epoch end."""
        if self.memory_profiler:
            self.memory_profiler.take_snapshot(f"epoch_{trainer.current_epoch}_end")
            
            # Perform memory analysis
            if self.enable_leak_detection and len(self.memory_profiler.memory_snapshots) > 10:
                analysis = self.memory_profiler.analyze_memory_usage()
                
                # Log memory analysis
                if 'leak_detection' in analysis:
                    leak_info = analysis['leak_detection']
                    
                    pl_module.log_dict({
                        'memory/cpu_trend_mb': leak_info.get('cpu_trend_mb_per_snapshot', 0),
                        'memory/gpu_trend_mb': leak_info.get('gpu_trend_mb_per_snapshot', 0),
                        'memory/cpu_leak_suspected': float(leak_info.get('cpu_leak_suspected', False)),
                        'memory/gpu_leak_suspected': float(leak_info.get('gpu_leak_suspected', False)),
                    }, on_step=False, on_epoch=True)
                    
                    # Warn about potential leaks
                    if leak_info.get('cpu_leak_suspected') or leak_info.get('gpu_leak_suspected'):
                        logger.warning("Potential memory leak detected - check memory usage patterns")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save final memory analysis when training ends."""
        if self.memory_profiler:
            self.memory_profiler.take_snapshot("training_end")
            
            # Save comprehensive analysis
            if self.save_snapshots:
                analysis_file = f"memory_analysis_{int(time.time())}.json"
                self.memory_profiler.save_profile(analysis_file)
                logger.info(f"Memory analysis saved to {analysis_file}")
            
            # Log final summary
            final_analysis = self.memory_profiler.analyze_memory_usage()
            logger.info(f"Final memory analysis: {final_analysis}")


# Export all classes for easy importing
__all__ = [
    'PerformanceMetrics',
    'GPUOptimizer', 
    'MemoryProfiler',
    'PerformanceMonitor',
    'PerformanceCallback',
    'MemoryProfilingCallback',
    'DistributedTrainingOptimizer'
]