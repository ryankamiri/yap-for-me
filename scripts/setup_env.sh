#!/bin/bash

set -e

ENV_NAME="yap"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Change to project root (one level up from scripts/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.11 -y
fi

echo "Activating environment: $ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing PyTorch with CUDA support via pip..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing other dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Environment setup complete!"
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"

