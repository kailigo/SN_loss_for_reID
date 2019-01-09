#!/bin/sh
#

number=0
DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
# EXP_ROOT=/export/reid_datasets/transformed_collection/self_train/analysis
# DATA_PATH=/export/reid_datasets/transformed_collection/Market1501

EXP_ROOT=/media/kai/6T/media/reid/transformed_collection/self_train/analysis
DATA_PATH=/media/kai/6T/media/reid/transformed_collection/Market1501


python script/experiment/train5.py \
-d '(1,)' \
--only_test false \
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
--run $number

exit


echo "========================================================"
echo "========================================================"

number=0
DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXP_ROOT=/export/reid_datasets/transformed_collection/self_train/analysis
DATA_PATH=/export/reid_datasets/transformed_collection/Market1501

python script/experiment/train5.py \
-d '(1,)' \
--only_test false \
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
--knnsoftmax_w 1 \
--run $number

echo "========================================================"
echo "========================================================"


number=0
DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXP_ROOT=/export/reid_datasets/transformed_collection/self_train/analysis
DATA_PATH=/export/reid_datasets/transformed_collection/Market1501

python script/experiment/train5.py \
-d '(1,)' \
--only_test false \
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
--knnsoftmax_w 10 \
--run $number

echo "========================================================"
echo "========================================================"


number=0
DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXP_ROOT=/export/reid_datasets/transformed_collection/self_train/analysis
DATA_PATH=/export/reid_datasets/transformed_collection/Market1501

python script/experiment/train5.py \
-d '(1,)' \
--only_test false \
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
--knnsoftmax_w 0.01 \
--run $number



echo "========================================================"
echo "========================================================"
number=0
DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXP_ROOT=/export/reid_datasets/transformed_collection/self_train/analysis
DATA_PATH=/export/reid_datasets/transformed_collection/Market1501

python script/experiment/train5.py \
-d '(1,)' \
--only_test false \
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
--knnsoftmax_w 0.001 \
--run $number



echo "========================================================"
echo "========================================================"
number=0
DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXP_ROOT=/export/reid_datasets/transformed_collection/self_train/analysis
DATA_PATH=/export/reid_datasets/transformed_collection/Market1501

python script/experiment/train5.py \
-d '(1,)' \
--only_test false \
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
--knnsoftmax_w 0.0001 \
--run $number

exit


