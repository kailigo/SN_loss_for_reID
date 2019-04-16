#!/bin/sh
#
number=0
DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval

EXP_ROOT=/media/kai/6T/media/reid/transformed_collection/self_train
DATA_PATH=/media/kai/6T/media/reid/transformed_collection/Market1501
# CKPT_DIR=/media/kai/6T/media/reid/transformed_collection/self_train/market_knnsoftmax_alpha50_k10_i800
EXP_DIR=/media/kai/6T/media/reid/transformed_collection/self_train/market_knnsoftmax_alpha50_k10_i800

python script/experiment/train5.py \
-d '(0,)' \
--exp_dir $EXP_DIR \
--only_test True \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_root $EXP_ROOT \
--steps_per_log 10 \
--epochs_per_val 5 \
--normalize_feature true \
--total_epochs 150 \
--exp_decay_at_epoch 76 \
--knnsoftmax_alpha 50 \
--knnsoftmax_k 10 \
--dataset_path $DATA_PATH \
--knnsoftmax_w 0.1 \
--run $number \
--ims_per_id 4 \
--ids_per_batch 32 \
--total_epochs 150 \
--resume True 