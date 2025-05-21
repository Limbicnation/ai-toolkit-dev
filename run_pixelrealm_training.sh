#!/bin/bash

# This script runs the PixelRealm LoRA training with memory optimizations

# Set PyTorch memory optimization variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow messages

# Check if dataset is empty and copy sample data if needed
DATASET_DIR="./dataset/PixelRealm_LoRA-Dataset_v2_2025-05-17"
if [ -z "$(ls -A $DATASET_DIR 2>/dev/null)" ]; then
  echo "Dataset directory is empty, copying sample data..."
  
  # Copy some sample data from existing pixel art samples
  if [ -d "./PixelRealm_LoRA-Dataset_v2_2025-05-17" ]; then
    cp -r ./PixelRealm_LoRA-Dataset_v2_2025-05-17/* $DATASET_DIR/
  elif [ -d "./pixel_art_training_dataset" ]; then
    cp -r ./pixel_art_training_dataset/* $DATASET_DIR/
  elif [ -d "./pixel-art-character_512" ]; then
    cp -r ./pixel-art-character_512/* $DATASET_DIR/
  else
    echo "Warning: No sample data found to copy into dataset directory."
    echo "Please add image files with corresponding .txt caption files to:"
    echo "$DATASET_DIR"
    exit 1
  fi
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
python run.py config/pixelrealm_lora/PixelRealm_LoRA-Dataset_v2_2025-05-17.yaml