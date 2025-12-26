#!/bin/bash
set -e

# # One log file for everything
# LOG_FILE="test_all_DC_DM.log"

# if [ -f "$LOG_FILE" ]; then
#   rm -f "$LOG_FILE"
# fi

# touch "$LOG_FILE"


# Datasets
DATASETS=(
  "CIFAR10_S_90"
  "Colored_FashionMNIST_foreground"
  "Colored_FashionMNIST_background"
  "Colored_MNIST_foreground"
  "Colored_MNIST_background"

  
)

  for dataset in "${DATASETS[@]}"; do

      echo "==================================================" 
      echo "START: dataset=${dataset}" 
      echo "Time: $(date)" 
      echo "==================================================" 

      python test_fairness.py \
        --dataset "$dataset" 

    #   echo "" 
    #   echo "END: dataset=${dataset}, ipc=${ipc}, metric=${metric}"
    #   echo "" 

    done

