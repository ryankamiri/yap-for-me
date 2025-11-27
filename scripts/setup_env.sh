#!/bin/bash

set -e

ENV_NAME="yap"

echo "Creating conda environment: $ENV_NAME"

conda create -n "$ENV_NAME" python=3.11 -y

echo "Activating environment: $ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing PyTorch with CUDA support via conda..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo "Installing other dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Environment setup complete!"
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"

