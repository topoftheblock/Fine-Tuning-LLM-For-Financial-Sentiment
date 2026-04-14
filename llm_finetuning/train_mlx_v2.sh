#!/bin/bash

echo "🚀 Starting Advanced V2 LoRA Training on Apple M4..."

# Advanced Hyperparameter Breakdown:
# --batch-size 2 : Keeps M4 memory usage safe while training
# --grad-accum 4 : Simulates a batch size of 8 (2 * 4) for smoother learning curves
# --lora-layers 32 : Targets more layers for deep domain adaptation
# --learning-rate 1e-5 : A smaller learning rate prevents overwriting English logic
# --iters 1500 : Enough steps to learn, but stops before severe overfitting

python -m mlx_lm.lora \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --train \
    --data ../data/processed \
    --batch-size 2 \
    --grad-accum 4 \
    --lora-layers 32 \
    --learning-rate 1e-5 \
    --iters 1500 \
    --val-batches 20 \
    --adapter-path ./v2_sentiment_adapters

echo "✅ V2 Fine-tuning complete!"