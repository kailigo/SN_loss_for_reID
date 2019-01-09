#!/bin/sh
#

DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/test
DATA_PATH=/export/reid_datasets/transformed_collection/Market1501
MODEL_WEIGHT_FILE=/export/reid_datasets/transformed_collection/self_train/market_best/ckpt.pth


python script/experiment/visualize_rank_list.py \
-d '(0,)' \
--num_queries 100 \
--rank_list_size 10 \
--dataset $DATASET_NAME \
--dataset_path $DATA_PATH \
--last_conv_stride $STRIDE \
--normalize_feature false \
--exp_dir $EXPERIMENT_DIRECTORY \
--ckpt_file $MODEL_WEIGHT_FILE

exit








# a dataset name (one of market1501, cuhk03, duke)
# stride, 1 or 2
# an experiment directory for saving testing log
# the path of the downloaded model_weight.pth


DATASET_NAME=market1501
STRIDE=1
TRAINVAL_OR_TRAIN=trainval
EXPERIMENT_DIRECTORY=/export/reid_datasets/transformed_collection/self_train/market_weight-e7/vis
MODEL_WEIGHT_FILE=/export/reid_datasets/transformed_collection/trained_models/market1501_stride1/model_weight.pth

# /export/reid_datasets/transformed_collection/self_train/market_weight-e7/ckpt.pth
＃/export/reid_datasets/transformed_collection/self_train/duke/ckpt.pth
#$$=/export/reid_datasets/transformed_collection/trained_models/duke_stride1/model_weight.pth
#/export/reid_datasets/transformed_collection/self_train/market_weight-e7/ckpt.pth

python script/experiment/visualize_rank_list.py \
-d '(0,)' \
--num_queries 16 \
--rank_list_size 10 \
--dataset $DATASET_NAME \
--last_conv_stride $STRIDE \
--normalize_feature false \
--exp_dir $EXPERIMENT_DIRECTORY \
--model_weight_file $MODEL_WEIGHT_FILE

exit






# --model_weight_file /export/reid_datasets/transformed_collection/self_train/market_weight_c1000_t1_x0/ckpt.pth
# --model_weight_file  /export/reid_datasets/transformed_collection/trained_models/duke_stride1/model_weight.pth

python script/experiment/train.py \
-d '(0,)' \
--only_test true \
--dataset market1501 \
--last_conv_stride 1 \
--normalize_feature false \
--exp_dir /export/reid_datasets/transformed_collection/trained_models/market1501_stride1_evaluation \
--model_weight_file /export/reid_datasets/transformed_collection/trained_models/market1501_stride1/model_weight.pth

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
