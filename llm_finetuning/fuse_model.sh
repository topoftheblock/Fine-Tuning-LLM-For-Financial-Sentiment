#!/bin/bash
echo "🧬 Fusing Qwen LoRA adapters..."

python -m mlx_lm.fuse \
    --model Qwen/Qwen2.5-3B-Instruct \
    --adapter-path ./qwen_adapters \
    --save-path ../inference/fused-qwen-finance