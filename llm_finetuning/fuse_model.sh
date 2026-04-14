#!/bin/bash
echo "🧬 Fusing Qwen 0.5B LoRA adapters..."

python -m mlx_lm.fuse \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter-path ./qwen_adapters \
    --save-path ../inference/fused-qwen-finance