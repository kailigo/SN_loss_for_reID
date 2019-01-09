#!/bin/sh
#


DATASET_NAME=cuhk03_OS
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






DATASET_NAME=cuhk03
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
--total_epochs 800 \
--exp_decay_at_epoch 76 \
--knnsoftmax_alpha 50 \
--knnsoftmax_k 10 \
--knnsoftmax_w 0.1 \

exit



DATASET_NAME=cuhk03
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXP_ROOT=/export/reid_datasets/transformed_collection/self_train

python script/experiment/train5.py \
-d '(0,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_root $EXP_ROOT \
--steps_per_log 10 \
--epochs_per_val 5 \
--normalize_feature false \
--total_epochs 800 \
--exp_decay_at_epoch 76 \
--knnsoftmax_alpha 50 \
--knnsoftmax_k 10 \
--knnsoftmax_w 0.1

exit



#!/bin/sh
#

# Specify
# - a dataset name (one of `['market1501', 'cuhk03', 'duke']`)
# - stride, `1` or `2`
# - training on `trainval` set or `train` set (for tuning parameters)
# - an experiment directory for saving training log
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
--knnsoftmax_w 0.1


exit
exit

DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market_knnsoftmax_alpha50_k30_i300

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
--total_epochs 300 \
--exp_decay_at_epoch 76 \
--knnsoftmax_alpha 50 \
--knnsoftmax_k 30


echo "========================================================"
echo "========================================================"
echo "========================================================"


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market_knnsoftmax_alpha50_k50_i300

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
--total_epochs 300 \
--exp_decay_at_epoch 76 \
--knnsoftmax_alpha 50 \
--knnsoftmax_k 50


echo "========================================================"
echo "========================================================"
echo "========================================================"


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market_knnsoftmax_alpha80_k100_i300

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
--total_epochs 300 \
--exp_decay_at_epoch 76 \
--knnsoftmax_alpha 80 \
--knnsoftmax_k 100



exit












# Specify
# - a dataset name (one of `['market1501', 'cuhk03', 'duke']`)
# - stride, `1` or `2`
# - training on `trainval` set or `train` set (for tuning parameters)
# - an experiment directory for saving training log
DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market_knnsoftmax_alpha20_k10_e1000

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
--exp_decay_at_epoch 500 \
--knnsoftmax_alpha 20 \
--knnsoftmax_k 10

exit


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market_weight_c100_t0_x0.1

python script/experiment/train1.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \
--center_loss_weight 100 \
--triplet_loss_weight 0.0 \
--xentrp_loss_weight 0.1

echo '================================================================================'
[mAP: 3.43%], [cmc1: 7.24%], [cmc5: 16.98%], [cmc10: 23.46%]
echo '================================================================================'
echo '================================================================================'


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market_weight_c1000_t0_x0.1

python script/experiment/train1.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \
--center_loss_weight 1000 \
--triplet_loss_weight 0.0 \
--xentrp_loss_weight 0.1


echo '================================================================================'
[mAP: 0.42%], [cmc1: 1.01%], [cmc5: 3.24%], [cmc10: 5.34%]
echo '================================================================================'
echo '================================================================================'


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market_weight_c1000_t1_x0

python script/experiment/train1.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \
--center_loss_weight 1000 \
--triplet_loss_weight 1.0 \
--xentrp_loss_weight 0.0

echo '================================================================================'
[mAP: 21.90%], [cmc1: 38.69%], [cmc5: 64.64%], [cmc10: 74.82%]
echo '================================================================================'
echo '================================================================================'


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market_weight_c1000_t10_x0

python script/experiment/train1.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \
--center_loss_weight 1000 \
--triplet_loss_weight 10 \
--xentrp_loss_weight 0.0

echo '================================================================================'
[mAP: 64.24%], [cmc1: 80.70%], [cmc5: 92.16%], [cmc10: 94.71%]
echo '================================================================================'
echo '================================================================================'


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market_weight_c1000_t100_x0

python script/experiment/train1.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \
--center_loss_weight 1000 \
--triplet_loss_weight 100 \
--xentrp_loss_weight 0.0

echo '================================================================================'
[mAP: 72.45%], [cmc1: 86.61%], [cmc5: 94.74%], [cmc10: 96.62%]
[mAP: 85.89%], [cmc1: 89.88%], [cmc5: 94.03%], [cmc10: 95.31%]
[mAP: 79.68%], [cmc1: 91.24%], [cmc5: 96.53%], [cmc10: 97.83%]
[mAP: 89.82%], [cmc1: 92.90%], [cmc5: 96.23%], [cmc10: 97.00%]
echo '================================================================================'
echo '================================================================================'


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market_weight_c1000_t1000_x0

python script/experiment/train1.py \
-d '(1,)' \
--only_test false \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--trainset_part $TRAINVAL_OR_TRAIN \
--exp_dir $EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5 \
--center_loss_weight 1000 \
--triplet_loss_weight 1000 \
--xentrp_loss_weight 0.0
