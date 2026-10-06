#!/bin/bash
# MoRE: Training script
# Usage: bash scripts/train.sh <dataset> [tag]
#
# Example:
#   bash scripts/train.sh eth my-eth-run
#   bash scripts/train.sh hotel

DATASET=${1:-eth}
TAG=${2:-more-ppo-${DATASET}}
CONFIG="configs/default.json"

echo "=== MoRE Training ==="
echo "Dataset: ${DATASET}"
echo "Tag: ${TAG}"
echo "Config: ${CONFIG}"
echo "===================="

accelerate launch -m more.train \
    --config_file ${CONFIG} \
    --dataset_name ${DATASET} \
    --tag ${TAG}
