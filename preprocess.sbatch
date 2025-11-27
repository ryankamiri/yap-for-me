#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --job-name=preprocess
#SBATCH --output=out/%j/log.log

cd $SLURM_SUBMIT_DIR

# load env
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate yap

# run script
export PYTHONUNBUFFERED=1
python3 -u preprocess_dataset.py \
  --config configs/config.yaml \
  --num_workers 28

