# CRAFT-DF Data Preparation Pipeline

This document provides comprehensive documentation for the CRAFT-DF data preparation script (`data_prep.py`), which processes video datasets for deepfake detection training.

## Overview

The data preparation pipeline integrates face detection, cropping, and DWT (Discrete Wavelet Transform) feature extraction into a unified system for processing large video datasets. It generates hierarchical data organization with comprehensive metadata management.

## Features

- **Batch Video Processing**: Efficient processing of large video datasets with progress tracking
- **Face Detection & Cropping**: OpenCV-based face detection with configurable confidence thresholds
- **DWT Feature Extraction**: Multi-level wavelet decomposition for frequency domain analysis
- **Hierarchical Organization**: Structured file organization (`real/fake/video_id/frame_xxx.npy`)
- **Metadata Management**: Comprehensive CSV metadata with schema validation
- **Error Handling**: Robust error handling with recovery mechanisms
- **Resume Capability**: Resume interrupted processing from checkpoints
- **Configurable Parameters**: Extensive configuration options via YAML or command line
- **Comprehensive Logging**: Detailed logging with multiple levels and file output

## Installation

Ensure you have the CRAFT-DF package installed with all dependencies:

```bash
pip install -r requirements.txt
```

Required dependencies:
- torch
- pytorch-lightning
- opencv-python
- PyWavelets
- scikit-learn
- pandas
- numpy
- tqdm
- PyYAML

## Quick Start

### Basic Usage

```bash
# Process videos with default settings
python data_prep.py --input_dir ./videos --output_dir ./processed --metadata_path ./metadata.csv

# Use configuration file
python data_prep.py --config config.yaml

# Generate configuration template
python data_prep.py --generate_config my_config.yaml
```

### Directory Structure

Your input directory should be organized as follows:

```
input_videos/
├── real/
│   ├── video1.mp4
│   ├── video2.avi
│   └── ...
└── fake/
    ├── video3.mp4
    ├── video4.mov
    └── ...
```

The script will generate the following output structure:

```
processed_data/
├── real/
│   ├── video1/
│   │   ├── frame_000000_face_00.npy
│   │   ├── frame_000000_face_00_dwt.npy
│   │   ├── frame_000001_face_00.npy
│   │   └── ...
│   └── video2/
│       └── ...
└── fake/
    ├── video3/
    └── video4/
        └── ...
```

## Configuration

### Configuration File

Use a YAML configuration file for complex setups:

```yaml
# Basic parameters
input_dir: ./input_videos
output_dir: ./processed_data
metadata_path: ./metadata.csv

# Processing parameters
max_faces_per_frame: 1
frame_skip: 1
max_frames_per_video: 1000

# Face detector settings
face_detector:
  min_detection_confidence: 0.7
  target_size: [224, 224]

# DWT processor settings
dwt_processor:
  wavelet: db4
  levels: 3
  mode: symmetric

# Logging
logging:
  level: INFO
  log_file: processing.log
```

### Command Line Options

All configuration options can be overridden via command line:

```bash
python data_prep.py \
  --input_dir ./videos \
  --output_dir ./processed \
  --metadata_path ./metadata.csv \
  --max_faces_per_frame 2 \
  --frame_skip 2 \
  --min_detection_confidence 0.8 \
  --wavelet db8 \
  --dwt_levels 4 \
  --log_level DEBUG \
  --log_file processing.log
```

## Parameters Reference

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `input_dir` | Directory containing input videos | `./input_videos` |
| `output_dir` | Directory for processed outputs | `./processed_data` |
| `metadata_path` | Path for metadata CSV file | `./metadata.csv` |

### Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_faces_per_frame` | int | 1 | Maximum faces to extract per frame |
| `frame_skip` | int | 1 | Process every Nth frame |
| `max_frames_per_video` | int | None | Maximum frames per video (None = unlimited) |
| `save_intermediate` | bool | true | Save intermediate results for recovery |
| `progress_bar` | bool | true | Show progress bar during processing |

### Face Detector Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_detection_confidence` | float | 0.7 | Minimum confidence for face detection (0.0-1.0) |
| `model_selection` | int | 0 | Model selection (0 = short-range, 1 = full-range) |
| `target_size` | list | [224, 224] | Target size for face crops [height, width] |

### DWT Processor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wavelet` | str | "db4" | Wavelet type (db4, db8, haar, bior2.2, etc.) |
| `levels` | int | 3 | Number of decomposition levels (1-6) |
| `mode` | str | "symmetric" | Boundary condition mode |

### Logging Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | str | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `log_file` | str | None | Log file path (None = console only) |

## Advanced Usage

### Resume Interrupted Processing

If processing is interrupted, you can resume from where it left off:

```bash
python data_prep.py --config config.yaml --resume
```

The script automatically saves intermediate results and can detect previously processed videos.

### Custom Video Extensions

Specify which video file types to process:

```yaml
video_extensions:
  - .mp4
  - .avi
  - .mov
  - .mkv
  - .wmv
```

### Memory Optimization

For large datasets, adjust processing parameters to manage memory usage:

```yaml
max_faces_per_frame: 1        # Reduce faces per frame
frame_skip: 2                 # Skip every other frame
max_frames_per_video: 500     # Limit frames per video
```

### Debugging

Enable debug logging for troubleshooting:

```bash
python data_prep.py --config config.yaml --log_level DEBUG --log_file debug.log
```

## Output Files

### Processed Data Files

- **Face crops**: `frame_XXXXXX_face_XX.npy` - Cropped face images (224x224x3)
- **DWT features**: `frame_XXXXXX_face_XX_dwt.npy` - DWT feature vectors

### Metadata Files

- **metadata.csv**: Main metadata file with all processing information
- **metadata.stats.json**: Processing statistics and summary
- **processing_summary.json**: Comprehensive processing report

### Metadata Schema

The metadata CSV contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | str | Video identifier (filename without extension) |
| `label` | str | Video label ("real" or "fake") |
| `label_numeric` | int | Numeric label (0 = real, 1 = fake) |
| `frame_number` | int | Frame number in video |
| `face_index` | int | Face index in frame |
| `face_confidence` | float | Face detection confidence score |
| `face_path` | str | Relative path to face crop file |
| `dwt_path` | str | Relative path to DWT features file |
| `dwt_feature_count` | int | Number of DWT features extracted |
| `processing_timestamp` | str | ISO timestamp of processing |
| `face_shape` | str | Face crop dimensions |
| `dwt_wavelet` | str | Wavelet type used |
| `dwt_levels` | int | DWT decomposition levels |
| `dwt_mode` | str | DWT boundary mode |

## Error Handling

The script includes comprehensive error handling:

- **Video Reading Errors**: Skips corrupted videos, logs warnings
- **Face Detection Failures**: Continues processing, logs failed frames
- **File I/O Errors**: Implements retry mechanisms
- **Memory Issues**: Provides guidance on parameter adjustment
- **Interruption Recovery**: Saves intermediate results for resume

## Performance Optimization

### Hardware Recommendations

- **CPU**: Multi-core processor for parallel video decoding
- **Memory**: 16GB+ RAM for large datasets
- **Storage**: SSD recommended for faster I/O operations

### Performance Tips

1. **Adjust frame_skip**: Process every 2nd or 3rd frame for faster processing
2. **Limit max_frames_per_video**: Set reasonable limits for very long videos
3. **Use SSD storage**: Significantly improves I/O performance
4. **Monitor memory usage**: Adjust batch sizes if memory issues occur

## Troubleshooting

### Common Issues

**Issue**: "No video files found"
- **Solution**: Check input directory structure and video file extensions

**Issue**: "Cannot determine label from path"
- **Solution**: Ensure videos are in "real" or "fake" subdirectories

**Issue**: "Face detection model loading failed"
- **Solution**: Verify OpenCV installation and model files

**Issue**: "Memory error during processing"
- **Solution**: Reduce max_faces_per_frame or increase frame_skip

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
python data_prep.py --config config.yaml --log_level DEBUG --log_file debug.log
```

## Examples

### Example 1: Basic Processing

```bash
python data_prep.py \
  --input_dir ./deepfake_videos \
  --output_dir ./processed_deepfakes \
  --metadata_path ./deepfake_metadata.csv
```

### Example 2: High-Quality Processing

```yaml
# config_hq.yaml
input_dir: ./high_quality_videos
output_dir: ./hq_processed
metadata_path: ./hq_metadata.csv

max_faces_per_frame: 2
frame_skip: 1
max_frames_per_video: 2000

face_detector:
  min_detection_confidence: 0.8
  target_size: [256, 256]

dwt_processor:
  wavelet: db8
  levels: 4
  mode: symmetric

logging:
  level: INFO
  log_file: hq_processing.log
```

```bash
python data_prep.py --config config_hq.yaml
```

### Example 3: Fast Processing for Large Datasets

```yaml
# config_fast.yaml
input_dir: ./large_dataset
output_dir: ./fast_processed
metadata_path: ./fast_metadata.csv

max_faces_per_frame: 1
frame_skip: 3
max_frames_per_video: 500

face_detector:
  min_detection_confidence: 0.6
  target_size: [224, 224]

dwt_processor:
  wavelet: db4
  levels: 2
  mode: symmetric
```

```bash
python data_prep.py --config config_fast.yaml --no_progress
```

## Integration with CRAFT-DF Training

The processed data can be directly used with the CRAFT-DF training pipeline:

```python
from craft_df.data.dataset import HierarchicalDeepfakeDataset

# Load processed dataset
dataset = HierarchicalDeepfakeDataset(
    metadata_path="./metadata.csv",
    transform=None
)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Enable debug logging for detailed error information
3. Verify input data format and directory structure
4. Ensure all dependencies are properly installed

## License

This data preparation pipeline is part of the CRAFT-DF deepfake detection system.