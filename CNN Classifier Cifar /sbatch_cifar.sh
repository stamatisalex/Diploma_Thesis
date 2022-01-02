#!/bin/bash
#SBATCH  --output=sbatch_cifar_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
source /scratch_net/petzi/salexandropo/anaconda3/etc/profile.d/conda.sh
conda activate experiments
python3 -u cifar10_tutorial_pytorch_version.py "$@"
