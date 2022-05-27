#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:4
#SBATCH  --mem=40G
source /scratch_net/petzi/salexandropo/anaconda3/etc/profile.d/conda.sh
conda activate train
PY_CMD="python3 -m torch.distributed.launch --nproc_per_node=4"
$PY_CMD tools/train.py --cfg experiments/acdc/seg_hrnet_acdc_w48_train_ohem_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
