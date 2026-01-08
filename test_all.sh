#!/usr/bin/env bash

IPC=100
ITER=15000
BATCH_REAL=256

DATASETS=(
  # "CIFAR10_S_90"
  # "Colored_FashionMNIST_foreground"
  # "Colored_FashionMNIST_background"
  # "Colored_MNIST_foreground"
  # "Colored_MNIST_background"
  "UTKface"
  # "BFFHQ"
)


for dataset in "${DATASETS[@]}"; do
  echo "=== Running dataset=${dataset}, ipc=${IPC} ==="
  python main_DM_MoFair.py \
    --dataset "${dataset}" \
    --ipc "${IPC}" \
    --Iteration "${ITER}" \
    --batch_real "${BATCH_REAL}" 
done

