#!/bin/bash
#SBATCH --job-name=medrag
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=results/rag_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medrag
export HF_HOME=$HOME/RAG/hf_cache HF_HUB_OFFLINE=1
cd ~/RAG

nvidia-smi --query-gpu=name,memory.total --format=csv
python src/rag.py
