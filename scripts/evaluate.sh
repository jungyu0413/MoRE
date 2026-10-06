#!/bin/bash
# MoRE: Evaluation script
# Usage: bash scripts/evaluate.sh <config> [checkpoint_path|""] [tag] [dataset]
#   empty checkpoint ("") -> best known weights for the dataset (weights/README.md)
#
# Example:
#   bash scripts/evaluate.sh configs/default.json checkpoint/my-run/best_model my-eval
#   bash scripts/evaluate.sh configs/default.json checkpoint/my-run/best_model my-eval hotel
#
# When [dataset] is omitted, dataset_name from the config file is used.

CONFIG=${1:-configs/default.json}
CHECKPOINT=${2}
TAG=${3:-eval}
DATASET=${4}

echo "=== MoRE Evaluation ==="
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT:-(best weights for the dataset)}"
echo "Dataset: ${DATASET:-(from config)}"
echo "========================"

accelerate launch -m more.evaluate \
    --config_file ${CONFIG} \
    ${CHECKPOINT:+--checkpoint ${CHECKPOINT}} \
    --tag ${TAG} \
    ${DATASET:+--dataset_name ${DATASET}}
