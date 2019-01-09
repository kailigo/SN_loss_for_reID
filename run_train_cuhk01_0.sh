#!/bin/sh
#

# su

for number in 0 1 2 3 4
do
echo "******************************"
echo "$number "
echo "******************************"


DATASET_NAME=cuhk01
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXP_ROOT=/export/reid_datasets/transformed_collection/self_train
# DATA_PATH=/export/reid_datasets/transformed_collection/CUHK03/labeled/official_split/$number
DATA_PATH=/export/reid_datasets/transformed_collection/cuhk01/test100/$number




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
--dataset_path $DATA_PATH \
--knnsoftmax_k 10 \
--knnsoftmax_w 0.1 \
--run $number

done

exit
