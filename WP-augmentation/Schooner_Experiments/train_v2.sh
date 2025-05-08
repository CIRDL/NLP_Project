#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=wp_roberta_%J_stdout.txt
#SBATCH --error=wp_roberta_%J_stderr.txt
#SBATCH --job-name=WP_RoBERTa
#SBATCH --mail-user=teddy.f.diallo-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/cs529304/project

source ~/nlp_env/bin/activate 

export WANDB_API_KEY=""
export HF_TOKEN=""
export WANDB_DISABLED=true

python train_v2.py

