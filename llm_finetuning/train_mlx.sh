#!/bin/bash
echo "🚀 Starting MLX LoRA Fine-Tuning for Qwen 0.5B on Apple M4 with W&B Tracking..."

python -m mlx_lm.lora \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --train \
    --data ../data/processed \
    --iters 1000 \
    --batch-size 8 \
    --lora-layers 16 \
    --adapter-path ./qwen_adapters \
    --report-to wandb  # <-- NEW: Enables automatic graphic generation

echo "✅ Qwen 0.5B fine-tuning complete!"