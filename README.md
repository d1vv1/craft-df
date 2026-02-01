# CRAFT-DF: Cross-Attentive Frequency-Temporal Disentanglement for Generalizable Deepfake Detection

A robust deepfake detection system that combines spatial and frequency domain analysis through a dual-stream architecture with cross-attention mechanisms.

## Features

- **Dual-Stream Architecture**: Combines spatial (MobileNetV2) and frequency (DWT) domain analysis
- **Cross-Attention Fusion**: Advanced attention mechanism for feature integration
- **Scalable Data Pipeline**: Hierarchical dataset management for massive video datasets
- **GPU Optimized**: Designed for high-performance GPUs like NVIDIA H100
- **Reproducible**: Comprehensive seed management and deterministic training

## Installation

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
./activate_env.sh
```

## Quick Start

```python
from craft_df.utils.config import load_default_config
from craft_df.utils.reproducibility import seed_everything

# Load configuration
config = load_default_config()

# Set up reproducibility
seed_everything(config.reproducibility.seed)

# Your training code here...
```

## Project Structure

```
craft_df/
├── craft_df/           # Main package
│   ├── data/          # Data processing modules
│   ├── models/        # Neural network architectures
│   ├── training/      # Training pipeline
│   └── utils/         # Utility functions
├── configs/           # Configuration files
├── tests/            # Test suite
└── requirements.txt  # Dependencies
```

## Configuration

The project uses YAML-based configuration management. See `configs/default.yaml` for available options.

## License

[Add your license here]