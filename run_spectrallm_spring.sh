#!/bin/bash

echo "======================================================================"
echo "SpectraLLM - Spring Dataset Experiment"
echo "======================================================================"
echo "Model: SpectraLLM (FreqAttn_NoValue架构)"
echo "Dataset: Spring 2018-2019"
echo "  - Training: 2018 spring + 2019 spring (except last 10 days)"
echo "  - Validation: 2019 spring last 10 days"
echo "  - Test: 2019 spring last 10 days (same as validation)"
echo "Metrics: Calculated on ORIGINAL SCALE"
echo "======================================================================"

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate llmtime

echo ""
echo ">>> Training SpectraLLM on Spring Dataset"
echo ""

python train_spectrallm_spring.py \
    --root_path ./dataset/spring_cleaned \
    --data_path train_spring_final.csv \
    --seq_len 24 \
    --pred_len 1 \
    --qwen2_path ./qwen2-0.5b-local \
    --qwen2_layers 6 \
    --adapter_dim 64 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --train_epochs 50 \
    --gpu 0 \
    --checkpoint_dir ./checkpoints/spectrallm_spring \
    2>&1 | tee logs/spectrallm_spring.log

echo ""
echo "======================================================================"
echo "✅ Experiment completed!"
echo "======================================================================"
echo "Results saved in:"
echo "  - logs/spectrallm_spring.log"
echo "  - checkpoints/spectrallm_spring/best_model.pth"
echo "======================================================================"
