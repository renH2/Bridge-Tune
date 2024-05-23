#!/bin/bash
saved_path=$1
model_path=$2
dataset=$3
cuda=$4

python train_bridge.py \
  --moco \
  --finetune \
  --model-path $saved_path \
  --gpu $cuda \
  --dataset $dataset \
  --epochs 50 \
  --resume "$saved_path/$model_path/current.pth"
python generate.py --load-path "./$saved_path/$model_path/current.pth" --dataset $dataset
bash scripts/node_classification/ours.sh "./$saved_path/$model_path/" 64 $dataset

