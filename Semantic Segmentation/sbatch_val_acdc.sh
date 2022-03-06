#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:4
#SBATCH  --mem=30G
source /scratch_net/petzi/salexandropo/anaconda3/etc/profile.d/conda.sh
conda activate train
python tools/test.py --cfg experiments/acdc/seg_hrnet_acdc_w48_train_ohem_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     TEST.MODEL_FILE output/acdc/seg_hrnet_acdc_w48_train_ohem_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/final_state.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
