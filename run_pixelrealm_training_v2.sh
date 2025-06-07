#!/bin/bash

# This script runs the PixelRealm LoRA training with memory optimizations

# Set PyTorch memory optimization variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow messages

# Check if dataset is empty and copy sample data if needed
DATASET_DIR="./dataset/PixelRealm_LoRA-Dataset_v5_Creative_2025-05-30"
if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A $DATASET_DIR 2>/dev/null)" ]; then
  echo "Error: Required dataset directory '$DATASET_DIR' does not exist or is empty."
  echo "Please ensure the PixelRealm_LoRA-Dataset_v5_Creative_2025-05-30 dataset is present."
  exit 1
fi

# Clear CUDA cache
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache()"

# Make sure system has enough resources
echo "Freeing up system resources..."
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true

# Install required packages
echo "Installing dependencies..."
pip install --upgrade bitsandbytes==0.45.5
pip install triton==3.2.0
pip install accelerate==0.28.0  # Make sure we have the latest accelerate package
pip install --upgrade optimum 

# Run the training with low memory settings
echo "Starting training with memory-optimized settings..."
python run.py config/pixelchar_refined_v7.yaml