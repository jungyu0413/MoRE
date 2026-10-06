#!/bin/bash
# MoRE: Precompute expert predictions before training
# Usage: bash scripts/prepare_experts.sh <config> [phase] [dataset]
#
# Example:
#   bash scripts/prepare_experts.sh configs/default.json train
#   bash scripts/prepare_experts.sh configs/default.json val
#   bash scripts/prepare_experts.sh configs/default.json train hotel
#
# When [dataset] is omitted, dataset_name from the config file is used.

CONFIG=${1:-configs/default.json}
PHASE=${2:-train}
DATASET=${3}

echo "=== MoRE: Prepare Expert Predictions ==="
echo "Config: ${CONFIG}"
echo "Phase: ${PHASE}"
echo "Dataset: ${DATASET:-(from config)}"
echo "========================================"

python -m more.prepare_experts \
    --config_file ${CONFIG} \
    --phase ${PHASE} \
    ${DATASET:+--dataset_name ${DATASET}}
