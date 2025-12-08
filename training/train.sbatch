#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=256GB
#SBATCH --time=08:00:00
#SBATCH --job-name=train-yap
#SBATCH --output=out/%j/log.log

cd $SLURM_SUBMIT_DIR    # ensure we start where sbatch was called

# load env
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate yap

# run script
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get number of GPUs allocated by SLURM
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU: use torchrun for DDP
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=29500
    
    torchrun --nproc_per_node=$NUM_GPUS train.py --config configs/config.yaml
else
    # Single GPU: run normally
    python3 -u train.py --config configs/config.yaml
fi
