# CRAFT-DF: Cross-Attentive Frequency-Temporal Disentanglement for Generalizable Deepfake Detection

A robust deepfake detection system that combines spatial and frequency domain analysis through a dual-stream architecture with cross-attention mechanisms. CRAFT-DF is designed for scalable training on massive video datasets with comprehensive experiment tracking and GPU optimization.

## Features

- **Dual-Stream Architecture**: Combines spatial (MobileNetV2) and frequency (DWT) domain analysis
- **Cross-Attention Fusion**: Advanced attention mechanism for feature integration  
- **Feature Disentanglement**: Adversarial training for domain generalization
- **Scalable Data Pipeline**: Hierarchical dataset management for massive video datasets
- **GPU Optimized**: Designed for high-performance GPUs like NVIDIA H100 with mixed precision
- **Reproducible**: Comprehensive seed management and deterministic training
- **Experiment Tracking**: Integrated Weights & Biases support with automatic checkpointing

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended: NVIDIA H100, A100, or RTX series)
- At least 16GB RAM (32GB+ recommended for large datasets)
- 50GB+ free disk space for processed data

### Setup Instructions

```bash
# Clone the repository
git clone <repository-url>
cd craft-df

# Create and activate virtual environment
python3 -m venv craft_df_env
source craft_df_env/bin/activate  # On Windows: craft_df_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Or use the provided activation script (macOS/Linux)
chmod +x activate_env.sh
./activate_env.sh
```

### Environment Variables

Before starting data ingestion or tracking workloads, you need to establish your environment secrets:

```bash
cp .env.example .env
```
Edit the `.env` file to include your **Kaggle** credentials (if downloading datasets automatically) and your **Weights & Biases** API key.

### Verify Installation

```bash
# Run basic tests to verify installation
python -m pytest tests/test_setup.py -v

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Basic Usage

```python
from craft_df.utils.config import load_default_config
from craft_df.utils.reproducibility import seed_everything
from craft_df.models.craft_df_model import CRAFTDFModel

# Load configuration
config = load_default_config()

# Set up reproducibility
seed_everything(config.reproducibility.seed)

# Initialize model
model = CRAFTDFModel(
    spatial_dim=config.model.spatial_dim,
    freq_dim=config.model.freq_dim,
    attention_heads=config.model.attention_heads
)

print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
```

### 2. End-to-End Pipeline

```python
# Complete pipeline example
from craft_df.training.training_pipeline import TrainingPipeline

# Initialize training pipeline
pipeline = TrainingPipeline(config_path="configs/default.yaml")

# Setup data loaders
train_loader, val_loader, test_loader = pipeline.setup_data_loaders()

# Setup model and trainer
model = pipeline.setup_model()
trainer = pipeline.setup_trainer()

# Start training
trainer.fit(model, train_loader, val_loader)
```

## Data Preparation

### Preparing Your Dataset

CRAFT-DF expects video files organized in the following structure:

```
input_videos/
├── real/
│   ├── video1.mp4
│   ├── video2.avi
│   └── ...
└── fake/
    ├── video1.mp4
    ├── video2.mov
    └── ...
```

### Running Data Preparation

The most efficient baseline to prepare and load data dynamically from Kaggle is via the included automation script. Make sure your `.env` contains valid Kaggle credentials:

```bash
# Automated dataset pipeline (fetches data, creates embeddings, builds CSV index)
python local_training/prepare_for_training.py
```
*Note: This generates a final `processed_dataset/` directory holding your arrays ready for training.*

**Manual Processing**
If you already have your own independent deepfake MP4 datasets:

```bash
# Basic data preparation
python data_prep.py \
    --input_dir ./input_videos \
    --output_dir ./processed_dataset \
    --metadata_path ./metadata.csv

# With custom configuration
python data_prep.py --config config_template.yaml

# Resume interrupted processing
python data_prep.py \
    --input_dir ./input_videos \
    --output_dir ./processed_dataset \
    --metadata_path ./metadata.csv \
    --resume
```

### Data Preparation Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_dir` | Directory containing input videos | Required |
| `--output_dir` | Directory for processed outputs | Required |
| `--metadata_path` | Path for metadata CSV file | Required |
| `--config` | Configuration file path | None |
| `--resume` | Resume interrupted processing | False |
| `--max_faces_per_frame` | Maximum faces to extract per frame | 1 |
| `--frame_skip` | Process every Nth frame | 1 |
| `--face_confidence` | Minimum face detection confidence | 0.7 |

### Output Structure

After processing, your data will be organized as:

```
processed_data/
├── real/
│   ├── video1/
│   │   ├── frame_001.npy  # Face crop + DWT coefficients
│   │   ├── frame_002.npy
│   │   └── ...
│   └── video2/
└── fake/
    └── ...
metadata.csv  # Registry of all processed files
```

## Training

### Basic Training

```bash
# Train with default configuration
python train.py --config configs/default.yaml

# Train with custom experiment name
python train.py \
    --config configs/default.yaml \
    --experiment my_experiment \
    --tags "baseline,mobilenet"

# Resume from checkpoint
python train.py \
    --config configs/default.yaml \
    --resume checkpoints/last.ckpt
```

### Advanced Training Options

```bash
# Debug mode (single batch, no logging)
python train.py --config configs/default.yaml --debug

# Distributed training (multi-GPU)
python train.py \
    --config configs/default.yaml \
    --devices 4 \
    --strategy ddp

# Custom data paths
python train.py \
    --config configs/default.yaml \
    --data_dir ./custom_data \
    --metadata_path ./custom_metadata.csv
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Configuration file path | Required |
| `--experiment` | Experiment name for logging | "default" |
| `--tags` | Comma-separated tags for experiment | None |
| `--resume` | Checkpoint path to resume from | None |
| `--debug` | Enable debug mode | False |
| `--devices` | Number of GPUs to use | 1 |
| `--strategy` | Training strategy (ddp, dp, etc.) | "auto" |
| `--data_dir` | Custom data directory | None |
| `--metadata_path` | Custom metadata file path | None |

## Configuration

### Configuration Files

CRAFT-DF uses YAML configuration files for all settings. The main configuration sections are:

#### Model Configuration

```yaml
model:
  spatial_backbone: "mobilenet_v2"     # Spatial stream backbone
  spatial_pretrained: true             # Use pretrained weights
  spatial_freeze_layers: 10            # Number of layers to freeze
  freq_dwt_levels: 3                   # DWT decomposition levels
  freq_wavelet: "db4"                  # Wavelet type
  attention_heads: 8                   # Number of attention heads
  attention_dim: 512                   # Attention dimension
  dropout_rate: 0.1                    # Dropout rate
```

#### Training Configuration

```yaml
training:
  learning_rate: 1e-4                  # Initial learning rate
  batch_size: 32                       # Batch size
  max_epochs: 100                      # Maximum training epochs
  num_workers: 4                       # Data loader workers
  pin_memory: true                     # Pin memory for GPU
  gradient_clip_val: 1.0               # Gradient clipping value
  accumulate_grad_batches: 1           # Gradient accumulation steps
  precision: 16                        # Mixed precision (16 or 32)
```

#### Data Configuration

```yaml
data:
  input_size: [224, 224]               # Input image size
  dwt_levels: 3                        # DWT decomposition levels
  wavelet_type: "db4"                  # Wavelet type
  face_confidence_threshold: 0.5       # Face detection threshold
  train_split: 0.7                     # Training split ratio
  val_split: 0.15                      # Validation split ratio
  test_split: 0.15                     # Test split ratio
```

#### Hardware Configuration

```yaml
hardware:
  accelerator: "gpu"                   # Accelerator type
  devices: 1                           # Number of devices
  strategy: "auto"                     # Training strategy
  sync_batchnorm: false               # Synchronize batch norm
```

### Environment Variables

You can override configuration values using environment variables:

```bash
export CRAFT_DF_BATCH_SIZE=64
export CRAFT_DF_LEARNING_RATE=2e-4
export CRAFT_DF_MAX_EPOCHS=200
```

## Project Structure

```
craft_df/
├── craft_df/                    # Main package
│   ├── __init__.py
│   ├── data/                    # Data processing modules
│   │   ├── __init__.py
│   │   ├── dataset.py          # PyTorch dataset implementation
│   │   ├── dwt_processing.py   # DWT feature extraction
│   │   ├── face_detection.py   # Face detection utilities
│   │   ├── transforms.py       # Data augmentation
│   │   └── video_processor.py  # Video processing pipeline
│   ├── models/                  # Neural network architectures
│   │   ├── __init__.py
│   │   ├── attention_visualization.py  # Attention analysis
│   │   ├── craft_df_model.py   # Main model architecture
│   │   ├── cross_attention.py  # Cross-attention mechanism
│   │   ├── feature_disentanglement.py  # Feature disentanglement
│   │   ├── frequency_stream.py # Frequency domain stream
│   │   └── spatial_stream.py   # Spatial domain stream
│   ├── training/               # Training pipeline
│   │   ├── __init__.py
│   │   ├── performance_monitor.py  # Performance monitoring
│   │   └── training_pipeline.py    # Main training logic
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── config.py          # Configuration management
│       └── reproducibility.py # Reproducibility utilities
├── configs/                    # Configuration files
│   ├── __init__.py
│   └── default.yaml           # Default configuration
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_*.py             # Individual test modules
│   └── conftest.py           # Test configuration
├── data_prep.py              # Data preparation script
├── train.py                  # Training script
├── config_template.yaml      # Data prep configuration template
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## API Reference

### Core Classes

#### CRAFTDFModel

Main model class implementing the dual-stream architecture.

```python
from craft_df.models.craft_df_model import CRAFTDFModel

model = CRAFTDFModel(
    spatial_dim=1280,      # Spatial feature dimension
    freq_dim=512,          # Frequency feature dimension
    attention_heads=8,     # Number of attention heads
    attention_dim=512,     # Attention dimension
    num_classes=2,         # Number of output classes
    dropout_rate=0.1       # Dropout rate
)

# Forward pass
spatial_features = torch.randn(batch_size, 1280)
freq_features = torch.randn(batch_size, 512)
output = model(spatial_features, freq_features)
```

#### HierarchicalDeepfakeDataset

PyTorch dataset for loading processed video data.

```python
from craft_df.data.dataset import HierarchicalDeepfakeDataset

dataset = HierarchicalDeepfakeDataset(
    metadata_path="metadata.csv",
    transform=None,           # Optional transforms
    cache_size=1000,         # Cache size for optimization
    validate_files=True      # Validate file existence
)

# Access samples
spatial_data, freq_data, label = dataset[0]
```

#### VideoProcessor

Video processing pipeline for data preparation.

```python
from craft_df.data.video_processor import VideoProcessor

processor = VideoProcessor(
    input_dir="./input_videos",
    output_dir="./processed_data",
    metadata_path="./metadata.csv",
    max_faces_per_frame=1,
    frame_skip=1
)

# Process videos
processor.process_all_videos()
```

### Utility Functions

#### Configuration Management

```python
from craft_df.utils.config import load_config, load_default_config

# Load custom configuration
config = load_config("path/to/config.yaml")

# Load default configuration
config = load_default_config()

# Access configuration values
batch_size = config.training.batch_size
learning_rate = config.training.learning_rate
```

#### Reproducibility

```python
from craft_df.utils.reproducibility import seed_everything

# Set all random seeds for reproducibility
seed_everything(42)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size in configuration: `training.batch_size: 16`
- Enable gradient accumulation: `training.accumulate_grad_batches: 2`
- Use mixed precision: `training.precision: 16`
- Clear GPU cache: `torch.cuda.empty_cache()`

```yaml
# Memory-optimized configuration
training:
  batch_size: 16
  accumulate_grad_batches: 2
  precision: 16
```

#### 2. Face Detection Failures

**Error**: `No faces detected in video`

**Solutions**:
- Lower face confidence threshold: `face_confidence_threshold: 0.3`
- Check video quality and resolution
- Verify video file format compatibility
- Use different face detection model

```yaml
# More permissive face detection
face_detector:
  min_detection_confidence: 0.3
  model_selection: 1  # Use full-range model
```

#### 3. DWT Processing Errors

**Error**: `ValueError: Invalid wavelet type` or `AssertionError: Expected 3D input`

**Solutions**:
- Use supported wavelet types: `db4`, `db8`, `haar`, `bior2.2`
- Ensure input images are RGB (3 channels)
- Check image dimensions are valid

```python
# Supported wavelets
supported_wavelets = ['db4', 'db8', 'haar', 'bior2.2', 'coif2']
```

#### 4. Dataset Loading Issues

**Error**: `FileNotFoundError` or `KeyError: 'spatial_path'`

**Solutions**:
- Verify metadata CSV has required columns: `spatial_path`, `frequency_path`, `label`
- Check file paths in metadata are correct
- Ensure processed data files exist
- Regenerate metadata if corrupted

```python
# Required metadata columns
required_columns = ['spatial_path', 'frequency_path', 'label', 'video_id']
```

#### 5. Training Convergence Issues

**Symptoms**: Loss not decreasing, accuracy stuck

**Solutions**:
- Adjust learning rate: `learning_rate: 5e-5`
- Enable learning rate scheduling
- Check data balance and augmentation
- Verify gradient flow through model

```yaml
# Learning rate scheduling
training:
  learning_rate: 1e-4
  lr_scheduler:
    type: "ReduceLROnPlateau"
    patience: 5
    factor: 0.5
```

#### 6. Attention Visualization Errors

**Error**: `RuntimeError: Attention weights have invalid shape`

**Solutions**:
- Ensure feature dimensions match model configuration
- Check batch size consistency
- Verify attention head configuration

#### 7. Multi-GPU Training Issues

**Error**: `RuntimeError: Expected all tensors to be on the same device`

**Solutions**:
- Use proper distributed strategy: `strategy: "ddp"`
- Ensure consistent device placement
- Check NCCL backend availability

```bash
# Multi-GPU training
python train.py \
    --config configs/default.yaml \
    --devices 4 \
    --strategy ddp
```

### Performance Debugging

#### Memory Profiling

```python
import torch.profiler

# Profile memory usage
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    # Your training code here
    pass

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

#### Performance Monitoring

```python
from craft_df.training.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_monitoring()

# Your training code here

stats = monitor.get_stats()
print(f"GPU Memory: {stats['gpu_memory_mb']:.1f} MB")
print(f"Throughput: {stats['samples_per_second']:.1f} samples/sec")
```

### Getting Help

1. **Check the logs**: Training logs contain detailed error information
2. **Run tests**: `python -m pytest tests/ -v` to identify issues
3. **Validate configuration**: Ensure YAML syntax is correct
4. **Check dependencies**: Verify all packages are installed correctly
5. **GPU diagnostics**: Run `nvidia-smi` to check GPU status

## Performance Optimization

### GPU Optimization

#### For NVIDIA H100/A100

```yaml
training:
  precision: 16                    # Use mixed precision
  batch_size: 64                   # Larger batch sizes
  num_workers: 8                   # More data loading workers
  pin_memory: true                 # Pin memory for faster transfers

hardware:
  accelerator: "gpu"
  devices: 1
  strategy: "auto"
  sync_batchnorm: false
```

#### For RTX Series

```yaml
training:
  precision: 16
  batch_size: 32                   # Moderate batch size
  accumulate_grad_batches: 2       # Gradient accumulation
  num_workers: 4

hardware:
  accelerator: "gpu"
  devices: 1
```

### Memory Optimization

```yaml
data:
  cache_size: 500                  # Reduce cache size
  lazy_loading: true               # Enable lazy loading
  
training:
  gradient_checkpointing: true     # Trade compute for memory
  batch_size: 16                   # Smaller batches
  accumulate_grad_batches: 4       # Maintain effective batch size
```

### Data Loading Optimization

```python
# Optimized data loader settings
train_loader = DataLoader(
    dataset,
    batch_size=config.training.batch_size,
    num_workers=config.training.num_workers,
    pin_memory=config.training.pin_memory,
    persistent_workers=True,         # Keep workers alive
    prefetch_factor=2,               # Prefetch batches
    drop_last=True                   # Consistent batch sizes
)
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd craft-df

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Run linting
flake8 craft_df/
black craft_df/
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Include unit tests for new features
- Maintain backward compatibility

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_models/ -v          # Model tests
python -m pytest tests/test_data/ -v            # Data tests
python -m pytest tests/test_integration.py -v   # Integration tests

# Run with coverage
python -m pytest tests/ --cov=craft_df --cov-report=html
```

## License

[Add your license here]

## Citation

If you use CRAFT-DF in your research, please cite:

```bibtex
@article{craft_df_2024,
  title={CRAFT-DF: Cross-Attentive Frequency-Temporal Disentanglement for Generalizable Deepfake Detection},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## Acknowledgments

- PyTorch Lightning team for the excellent training framework
- OpenCV community for computer vision utilities
- PyWavelets developers for wavelet transform implementation
- Weights & Biases for experiment tracking platform