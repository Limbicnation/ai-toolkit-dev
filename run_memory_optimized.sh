#!/bin/bash

# Memory optimization script for FLUX training
echo "Setting up memory optimization environment..."

# Memory management variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1 
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow messages

# Clear CUDA cache
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache()"

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