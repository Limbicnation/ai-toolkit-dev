#!/bin/bash

# Memory optimization script for FLUX training
echo "Setting up memory optimization environment..."

# Set PyTorch memory allocator configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Enable garbage collection
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Clear GPU cache before running
python -c "import torch; torch.cuda.empty_cache()"

export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1 
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow messages

# Free up system resources
echo "Freeing system resources..."
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true

# Show GPU memory status
echo "GPU status before training:"
nvidia-smi

# Run with memory optimization
echo "Running training with optimized memory configuration..."
python run.py config/pixelchar_refined_v2.yaml