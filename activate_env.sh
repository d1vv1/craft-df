#!/bin/bash
# Activation script for CRAFT-DF development environment

echo "🚀 Activating CRAFT-DF development environment..."

# Activate virtual environment
source craft_df_env/bin/activate

# Verify installation
echo "📦 Checking installation..."
python -c "import craft_df; print(f'✅ CRAFT-DF v{craft_df.__version__} ready!')"

# Show environment info
echo "🐍 Python: $(python --version)"
echo "📍 Virtual env: $VIRTUAL_ENV"
echo "🎯 Ready for development!"

# Optional: Set environment variables for reproducibility
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo ""
echo "💡 Usage tips:"
echo "  - Run tests: pytest tests/ -v"
echo "  - Format code: black craft_df/"
echo "  - Check types: mypy craft_df/"
echo "  - Install in dev mode: pip install -e ."
echo ""