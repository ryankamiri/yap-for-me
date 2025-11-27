#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=yap
#SBATCH --output=out/%j/logs.log

cd $SLURM_SUBMIT_DIR    # ensure we start where sbatch was called

# load env
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate comlrl

# run script
export PYTHONUNBUFFERED=1
python3 -u train.py \
  --config configs/magrpo_config.yaml
