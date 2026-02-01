"""
Configuration management utilities for CRAFT-DF.

This module provides utilities for loading and managing configuration files
using YAML format with validation and type checking.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from omegaconf import DictConfig, OmegaConf


class ConfigManager:
    """
    Configuration manager for CRAFT-DF project.
    
    Handles loading, validation, and management of YAML configuration files
    with support for nested configurations and environment variable substitution.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files.
                       Defaults to 'configs' in project root.
        """
        if config_dir is None:
            # Default to configs directory in project root
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "configs"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_path: Union[str, Path]) -> DictConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file (relative to config_dir or absolute)
            
        Returns:
            DictConfig: Loaded configuration with OmegaConf support
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_path)
        
        # Handle relative paths
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            # Convert to OmegaConf for advanced features
            config = OmegaConf.create(config_dict)
            
            # Resolve environment variables and interpolations
            config = OmegaConf.to_container(config, resolve=True)
            config = OmegaConf.create(config)
            
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    
    def save_config(self, config: Union[Dict[str, Any], DictConfig], 
                   config_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary or DictConfig to save
            config_path: Path where to save the configuration
        """
        config_path = Path(config_path)
        
        # Handle relative paths
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
        
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert DictConfig to regular dict if needed
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    def merge_configs(self, *config_paths: Union[str, Path]) -> DictConfig:
        """
        Merge multiple configuration files.
        
        Args:
            *config_paths: Paths to configuration files to merge
            
        Returns:
            DictConfig: Merged configuration (later configs override earlier ones)
        """
        merged_config = OmegaConf.create({})
        
        for config_path in config_paths:
            config = self.load_config(config_path)
            merged_config = OmegaConf.merge(merged_config, config)
        
        return merged_config
    
    def validate_config(self, config: DictConfig, schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against a schema.
        
        Args:
            config: Configuration to validate
            schema: Schema dictionary defining required fields and types
            
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration doesn't match schema
        """
        def _validate_recursive(cfg: Any, sch: Any, path: str = "") -> None:
            if isinstance(sch, dict):
                if not isinstance(cfg, (dict, DictConfig)):
                    raise ValueError(f"Expected dict at {path}, got {type(cfg)}")
                
                for key, value_schema in sch.items():
                    if key not in cfg:
                        raise ValueError(f"Missing required key: {path}.{key}")
                    _validate_recursive(cfg[key], value_schema, f"{path}.{key}")
            
            elif isinstance(sch, type):
                if not isinstance(cfg, sch):
                    raise ValueError(f"Expected {sch.__name__} at {path}, got {type(cfg)}")
        
        try:
            _validate_recursive(config, schema)
            return True
        except ValueError:
            raise


def load_default_config() -> DictConfig:
    """
    Load default configuration for CRAFT-DF.
    
    Returns:
        DictConfig: Default configuration
    """
    config_manager = ConfigManager()
    
    # Try to load default config, create if doesn't exist
    default_config_path = "default.yaml"
    
    try:
        return config_manager.load_config(default_config_path)
    except FileNotFoundError:
        # Create default configuration
        default_config = {
            "model": {
                "spatial_backbone": "mobilenet_v2",
                "spatial_pretrained": True,
                "spatial_freeze_layers": 10,
                "freq_dwt_levels": 3,
                "freq_wavelet": "db4",
                "attention_heads": 8,
                "attention_dim": 512,
                "dropout_rate": 0.1
            },
            "training": {
                "learning_rate": 1e-4,
                "batch_size": 32,
                "max_epochs": 100,
                "num_workers": 4,
                "pin_memory": True,
                "gradient_clip_val": 1.0
            },
            "data": {
                "input_size": [224, 224],
                "dwt_levels": 3,
                "wavelet_type": "db4",
                "face_confidence_threshold": 0.5
            },
            "logging": {
                "project_name": "craft-df",
                "experiment_name": "default",
                "log_every_n_steps": 50,
                "save_top_k": 3
            },
            "reproducibility": {
                "seed": 42,
                "deterministic": True,
                "benchmark": False
            }
        }
        
        config_manager.save_config(default_config, default_config_path)
        return OmegaConf.create(default_config)