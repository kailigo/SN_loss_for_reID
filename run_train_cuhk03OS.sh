#!/bin/sh
#

# su


mv /export/reid_datasets/transformed_collection/CUHK03/labeled/official_split/0 /export/reid_datasets/transformed_collection/cuhk01/test100/split/done_0

for number in 1 2 3 4
do
echo "******************************"
echo "$number "
echo "******************************"

# mv /export/reid_datasets/transformed_collection/CUHK03/labeled/official_split/$number /export/reid_datasets/transformed_collection/CUHK03/labeled/official_split/0

DATASET_NAME=cuhk03_OS
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXP_ROOT=/export/reid_datasets/transformed_collection/self_train
DATA_PATH=/export/reid_datasets/transformed_collection/CUHK03/labeled/official_split/$number


python script/experiment/train5.py \
-d '(0,)' \
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

# mv /export/reid_datasets/transformed_collection/CUHK03/labeled/official_split/0 /export/reid_datasets/transformed_collection/CUHK03/labeled/official_split/done_$number

done

exit


for number in {0..10..100}
do

echo "$number "

done


exit


DATASET_NAME=cuhk01
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXP_ROOT=/export/reid_datasets/transformed_collection/self_train

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
--knnsoftmax_w 0.1 \
--base_lr 1e-3



exit





DATASET_NAME=cuhk01
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXP_ROOT=/export/reid_datasets/transformed_collection/self_train

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
--knnsoftmax_w 0.1


exit







DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market

python script/experiment/train.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \

exit



# Specify
# - a dataset name (one of `['market1501', 'cuhk03', 'duke']`)
# - stride, `1` or `2`
# - training on `trainval` set or `train` set (for tuning parameters)
# - an experiment directory for saving training log

# DATASET_NAME=market1501
# STRIDE=1
# TRAINVAL_OR_TRAIN=trainval

# python script/experiment/train5.py \
# -d '(1,)' \
# --only_test false \
# --dataset $DATASET_NAME \
# --last_conv_stride $STRIDE \
# --normalize_feature false \
# --trainset_part $TRAINVAL_OR_TRAIN \
# --steps_per_log 10 \
# --epochs_per_val 5 \
# --normalize_feature true \
# --total_epochs 200 \
# --knnsoftmax_alpha 50 \
# --knnsoftmax_k 10 \
# --exp_root /export/reid_datasets/transformed_collection/self_train



DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXP_ROOT=/export/reid_datasets/transformed_collection/self_train

python script/experiment/train5.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_root $EXP_ROOT \
--steps_per_log 10 \
--epochs_per_val 5 \
--normalize_feature true \
--total_epochs 800 \
--exp_decay_at_epoch 76 \
--knnsoftmax_alpha 50 \
--knnsoftmax_k 10 \
--knnsoftmax_w 0.01


exit



DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval

python script/experiment/train4.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--steps_per_log 10 \
--epochs_per_val 5 \
--normalize_feature true \
--total_epochs 200 \
--lr_decay_type staircase \
--knnsoftmax_alpha 50 \
--knnsoftmax_k 10 \
--exp_root /media/kai/reid/transformed_collection/self_train


echo "========================================================"
echo "========================================================"
echo "========================================================"


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval

python script/experiment/train4.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \
--normalize_feature true \
--total_epochs 500 \
--knnsoftmax_alpha 50 \
--exp_root /media/kai/reid/transformed_collection/self_train \
--knnsoftmax_k 10


echo "========================================================"
echo "========================================================"
echo "========================================================"


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval

python script/experiment/train4.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \
--normalize_feature true \
--total_epochs 800 \
--exp_decay_at_epoch 76 \
--knnsoftmax_alpha 50 \
--exp_root /media/kai/reid/transformed_collection/self_train \
--knnsoftmax_k 10


echo "========================================================"
echo "========================================================"
echo "========================================================"


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval

python script/experiment/train4.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \
--normalize_feature true \
--total_epochs 1000 \
--exp_decay_at_epoch 76 \
--knnsoftmax_alpha 50 \
--exp_root /media/kai/reid/transformed_collection/self_train \
--knnsoftmax_k 10



exit












# Specify
# - a dataset name (one of `['market1501', 'cuhk03', 'duke']`)
# - stride, `1` or `2`
# - training on `trainval` set or `train` set (for tuning parameters)
# - an experiment directory for saving training log


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market_weight-e7

# /export/reid_datasets/transformed_collection/self_train/duke

python script/experiment/train.py \
-d '(0,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \

exit




DATASET_NAME=duke
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/duke

python script/experiment/train.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \
--center_loss_weight 0

exit


DATASET_NAME=duke
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/media/kai/reid/transformed_collection/self_train/duke


python script/experiment/train.py \
-d '(3,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5





DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market

python script/experiment/train.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \
--center_loss_weight 0
