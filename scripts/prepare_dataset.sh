#!/bin/bash
# MoRE: Build the preprocessed prompt/answer datasets from raw ETH/UCY files
# Usage: bash scripts/prepare_dataset.sh <config> [dataset...]
#
# Example:
#   bash scripts/prepare_dataset.sh configs/default.json
#   bash scripts/prepare_dataset.sh configs/default.json eth hotel univ zara1 zara2
#
# When no dataset is given, dataset_name from the config file is used.
# Writes <dataset_path>/preprocessed/*.json for the train, val and test splits.

CONFIG=${1:-configs/default.json}
shift
DATASETS="$@"

echo "=== MoRE: Prepare Datasets ==="
echo "Config: ${CONFIG}"
echo "Datasets: ${DATASETS:-(from config)}"
echo "=============================="

python -m more.prepare_dataset \
    --config_file ${CONFIG} \
    ${DATASETS:+--dataset_name ${DATASETS}}
