#!/bin/bash
# 时序大模型Adapter微调实验 - 长期预测 (seq=128, pred=96)
# 与SpectraLLM对比

GPU=${1:-1}
EPOCHS=50

echo "=========================================="
echo "TSFM Adapter Experiments - Long-term"
echo "seq_len=128, pred_len=96, epochs=$EPOCHS"
echo "GPU: $GPU"
echo "=========================================="

cd /home/lm/FreGPT/FreGPT_V8_Training_Only/SpectraLLM

# Transformer Baseline
echo ""
echo ">>> Training Transformer Baseline..."
CUDA_VISIBLE_DEVICES=$GPU python experiments/train_tsfm_adapter.py \
    --model_type transformer \
    --seq_len 128 --pred_len 96 \
    --train_epochs $EPOCHS \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --gpu 0 \
    --checkpoint_dir ./checkpoints/tsfm_long 2>&1 | tee logs/tsfm_transformer_long.log

# Timer
echo ""
echo ">>> Training Timer Adapter..."
CUDA_VISIBLE_DEVICES=$GPU python experiments/train_tsfm_adapter.py \
    --model_type timer \
    --model_path ./pretrained_models/timer-base-84m \
    --seq_len 128 --pred_len 96 \
    --train_epochs $EPOCHS \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --gpu 0 \
    --checkpoint_dir ./checkpoints/tsfm_long 2>&1 | tee logs/tsfm_timer_long.log

# Chronos
echo ""
echo ">>> Training Chronos Adapter..."
CUDA_VISIBLE_DEVICES=$GPU python experiments/train_tsfm_adapter.py \
    --model_type chronos \
    --model_path ./pretrained_models/chronos-t5-base \
    --seq_len 128 --pred_len 96 \
    --train_epochs $EPOCHS \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --gpu 0 \
    --checkpoint_dir ./checkpoints/tsfm_long 2>&1 | tee logs/tsfm_chronos_long.log

# Sundial
echo ""
echo ">>> Training Sundial Adapter..."
CUDA_VISIBLE_DEVICES=$GPU python experiments/train_tsfm_adapter.py \
    --model_type sundial \
    --model_path ./pretrained_models/sundial-base-128m \
    --seq_len 128 --pred_len 96 \
    --train_epochs $EPOCHS \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --gpu 0 \
    --checkpoint_dir ./checkpoints/tsfm_long 2>&1 | tee logs/tsfm_sundial_long.log

echo ""
echo "=========================================="
echo "All long-term experiments completed!"
echo "=========================================="
