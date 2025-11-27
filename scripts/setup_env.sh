#!/bin/bash

set -e

ENV_NAME="yap"

echo "Creating conda environment: $ENV_NAME"

conda create -n "$ENV_NAME" python=3.11 -y

echo "Activating environment: $ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing PyTorch with CUDA 12.3 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123

echo "Installing other dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Environment setup complete!"
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"

