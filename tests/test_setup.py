"""
Test basic project setup and imports.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_package_imports():
    """Test that main package modules can be imported."""
    try:
        import craft_df
        assert craft_df.__version__ == "0.1.0"
    except ImportError as e:
        pytest.fail(f"Failed to import craft_df: {e}")


def test_config_utilities():
    """Test configuration management utilities."""
    try:
        from craft_df.utils.config import ConfigManager, load_default_config
        
        # Test ConfigManager instantiation
        config_manager = ConfigManager()
        assert config_manager.config_dir.exists()
        
        # Test default config loading
        config = load_default_config()
        assert config is not None
        assert "model" in config
        assert "training" in config
        
    except ImportError as e:
        pytest.fail(f"Failed to import config utilities: {e}")


def test_reproducibility_utilities():
    """Test reproducibility utilities."""
    try:
        from craft_df.utils.reproducibility import seed_everything, get_reproducibility_info
        
        # Test seed_everything function
        seed = seed_everything(42)
        assert seed == 42
        
        # Test reproducibility info
        info = get_reproducibility_info()
        assert isinstance(info, dict)
        assert "torch_version" in info
        
    except ImportError as e:
        pytest.fail(f"Failed to import reproducibility utilities: {e}")


def test_project_structure():
    """Test that required project directories exist."""
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        "craft_df",
        "craft_df/data",
        "craft_df/models", 
        "craft_df/training",
        "craft_df/utils",
        "configs",
        "tests"
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Required directory {dir_path} does not exist"
        
        # Check for __init__.py in Python packages
        if dir_path.startswith("craft_df"):
            init_file = full_path / "__init__.py"
            assert init_file.exists(), f"Missing __init__.py in {dir_path}"


def test_requirements_file():
    """Test that requirements.txt exists and contains expected dependencies."""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    assert requirements_file.exists(), "requirements.txt file does not exist"
    
    with open(requirements_file, 'r') as f:
        requirements_content = f.read()
    
    # Check for key dependencies
    expected_deps = [
        "torch",
        "pytorch-lightning", 
        "opencv-python",
        "PyWavelets",
        "scikit-learn",
        "PyYAML"
    ]
    
    for dep in expected_deps:
        assert dep in requirements_content, f"Missing dependency: {dep}"


if __name__ == "__main__":
    pytest.main([__file__])