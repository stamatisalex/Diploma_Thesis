#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
source /scratch_net/petzi/salexandropo/anaconda3/etc/profile.d/conda.sh
conda activate train
python3 test.py