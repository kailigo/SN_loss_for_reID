#!/bin/sh
#


# a dataset name (one of market1501, cuhk03, duke)
# stride, 1 or 2
# an experiment directory for saving testing log
# the path of the downloaded model_weight.pth




number=0

DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval

EXP_ROOT=/export/reid_datasets/transformed_collection/self_train
DATA_PATH=/export/reid_datasets/Market-1501-v15.09.15/testing_with_distractors/raw_100k
CKPT_DIR=/export/reid_datasets/transformed_collection/self_train/market_best

python script/experiment/train_finetune.py \
-d '(0,)' \
--only_test true \
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
--ckpt_dir $CKPT_DIR \
--resume True \
--total_epochs 150


number=0

DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval

EXP_ROOT=/export/reid_datasets/transformed_collection/self_train
DATA_PATH=/export/reid_datasets/Market-1501-v15.09.15/testing_with_distractors/raw_200k
CKPT_DIR=/export/reid_datasets/transformed_collection/self_train/market_best

python script/experiment/train_finetune.py \
-d '(0,)' \
--only_test true \
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
--ckpt_dir $CKPT_DIR \
--resume True \
--total_epochs 150


number=0

DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval

EXP_ROOT=/export/reid_datasets/transformed_collection/self_train
DATA_PATH=/export/reid_datasets/Market-1501-v15.09.15/testing_with_distractors/raw_300k
CKPT_DIR=/export/reid_datasets/transformed_collection/self_train/market_best

python script/experiment/train_finetune.py \
-d '(0,)' \
--only_test true \
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
--ckpt_dir $CKPT_DIR \
--resume True \
--total_epochs 150


number=0

DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval

EXP_ROOT=/export/reid_datasets/transformed_collection/self_train
DATA_PATH=/export/reid_datasets/Market-1501-v15.09.15/testing_with_distractors/raw_400k
CKPT_DIR=/export/reid_datasets/transformed_collection/self_train/market_best

python script/experiment/train_finetune.py \
-d '(0,)' \
--only_test true \
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
--ckpt_dir $CKPT_DIR \
--resume True \
--total_epochs 150

number=0

DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval

EXP_ROOT=/export/reid_datasets/transformed_collection/self_train
DATA_PATH=/export/reid_datasets/Market-1501-v15.09.15/testing_with_distractors/raw_500k
CKPT_DIR=/export/reid_datasets/transformed_collection/self_train/market_best

python script/experiment/train_finetune.py \
-d '(0,)' \
--only_test true \
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
--ckpt_dir $CKPT_DIR \
--resume True \
--total_epochs 150

exit








python script/experiment/train.py \
-d '(0,)' \
--only_test true \
--dataset market1501 \
--last_conv_stride 1 \
--normalize_feature false \
--exp_dir /export/reid_datasets/transformed_collection/trained_models/market1501_stride1_evaluation \
--model_weight_file /export/reid_datasets/transformed_collection/trained_models/market1501_stride1/model_weight.pth

# --model_weight_file /export/reid_datasets/transformed_collection/self_train/market_weight_c1000_t1_x0/ckpt.pth


# --model_weight_file  /export/reid_datasets/transformed_collection/trained_models/duke_stride1/model_weight.pth

exit


python script/experiment/train.py \
-d '(0,)' \
--only_test true \
--dataset cuhk03 \
--last_conv_stride 1 \
--normalize_feature false \
--exp_dir /export/reid_datasets/transformed_collection/trained_models/cuhk03_stride1_evaluation \
--model_weight_file  /export/reid_datasets/transformed_collection/trained_models/cuhk03_stride1/model_weight.pth

exit



python script/experiment/train.py \
-d '(0,)' \
--only_test true \
--dataset market1501 \
--last_conv_stride 1 \
--normalize_feature false \
--exp_dir /export/reid_datasets/transformed_collection/trained_models/market1501_stride1_evaluation \
--model_weight_file  /export/reid_datasets/transformed_collection/trained_models/market1501_stride1/model_weight.pth
