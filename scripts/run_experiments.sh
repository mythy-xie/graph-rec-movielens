#!/bin/bash


set -e

DATASET="100k"
EPOCHS=50
LR=0.01
COLD_START_RATIO=0.1
DEVICE="auto"

echo "========================================================================"
echo "Start the automated comparison experiment | Dataset: ML-${DATASET} | Cold start ratio: ${COLD_START_RATIO}"
echo "========================================================================"

for MODEL in SAGE GCN GGNN
do
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Training: ${MODEL} ..."
    echo "------------------------------------------------------------------------"

    uv run src/trainer_plus.py \
        --dataset ${DATASET} \
        --model ${MODEL} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --cold_start_ratio ${COLD_START_RATIO} \
        --device ${DEVICE}

    echo "${MODEL} Training and evaluation complete!"
    sleep 2
done

echo ""
echo "========================================================================"
echo "All experiments have been completed! Please go to the logs/ directory to view the detailed results for each model."
echo "========================================================================"