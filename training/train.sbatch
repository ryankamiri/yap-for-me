#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --job-name=train-yap
#SBATCH --output=out/%j/log.log
#SBATCH --signal=SIGTERM@60

cd $SLURM_SUBMIT_DIR    # ensure we start where sbatch was called

# load env
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate yap

# run script
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check if --resume argument was passed
RESUME_ARG=""
if [ "$1" == "--resume" ] && [ -n "$2" ]; then
    RESUME_ARG="--resume $2"
    echo "Resuming from job: $2"
fi

python3 -u train.py \
  --config configs/config.yaml \
  $RESUME_ARG
