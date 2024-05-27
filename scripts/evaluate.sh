#!/bin/bash
saved_path=$1
model_path=$2
dataset=$3
cuda=$4

python generate.py --load-path "./$saved_path/$model_path/current.pth" --dataset $dataset
bash scripts/node_classification/ours.sh "./$saved_path/$model_path/" 64 $dataset

