#!/bin/bash
echo "🚀 Starting MLX LoRA Fine-Tuning for Qwen on Apple M4..."

python -m mlx_lm.lora \
    --model Qwen/Qwen2.5-3B-Instruct \
    --train \
    --data ../data/processed \
    --iters 1000 \
    --batch-size 4 \
    --lora-layers 16 \
    --adapter-path ./qwen_adapters

echo "✅ Qwen fine-tuning complete!"