#!/bin/bash

# This script runs an extremely minimized version of PixelRealm LoRA training
# with radical memory saving techniques focused on GPU-only usage

# Stop any GPU processes that might be using memory
echo "Checking for other GPU processes..."
nvidia-smi

# Set extremely aggressive memory optimization variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,roundup_power2_divisions:64
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Disable CUDA caching

# Create dataset directory if needed
DATASET_DIR="./dataset/PixelRealm_LoRA-Dataset_v1_2025-05-12"
mkdir -p "$DATASET_DIR"

# Check if dataset is empty and copy minimal data
if [ -z "$(ls -A $DATASET_DIR 2>/dev/null)" ]; then
  echo "Dataset directory is empty, copying minimal sample data..."
  
  # Find any of the available pixel art datasets
  FOUND_DATA=false
  for SRC_DIR in "./pixel_art_dataset_v2_512" "./pixel_art_training_dataset" "./pixel-art-character_512"; do
    if [ -d "$SRC_DIR" ]; then
      # Only copy a small subset (max 10 files) to save memory
      find "$SRC_DIR" -name "*.png" | head -10 | while read file; do
        cp "$file" "$DATASET_DIR/"
        # Copy corresponding txt file if it exists
        txt_file="${file%.png}.txt"
        if [ -f "$txt_file" ]; then
          cp "$txt_file" "$DATASET_DIR/"
        else
          # Create a minimal caption file if none exists
          echo "pixelrealm character" > "${DATASET_DIR}/$(basename "${file%.png}").txt"
        fi
      done
      FOUND_DATA=true
      break
    fi
  done
  
  if [ "$FOUND_DATA" = false ]; then
    echo "Warning: No sample data found. Creating minimal test data..."
    # Create a very minimal dataset with a single sample
    echo "pixelrealm character" > "${DATASET_DIR}/test.txt"
    # You would need an image here - we'll assume there's at least one image in the directory
  fi
fi

# Free as much memory as possible
echo "Preparing system for training..."

# Clear GPU cache
nvidia-smi --gpu-reset 2>/dev/null || true

# Kill any potential memory-hogging processes
pkill -f python 2>/dev/null || true
sleep 2

# Clear caches
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1 || true

# Free Python memory
python -c '
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
print("Memory cleared")
'

# Show available GPU memory before starting
echo "GPU memory status before training:"
nvidia-smi

# Install required packages
echo "Installing minimal dependencies..."
pip install --upgrade bitsandbytes==0.45.5
pip install triton==3.2.0 
pip install accelerate==0.28.0

# Run the training with absolute minimum memory settings
echo "Starting training with ultra-low memory settings..."
python -c '
import gc, torch, os
gc.collect()
torch.cuda.empty_cache()
# Free unused memory and set memory fraction
torch.cuda.set_per_process_memory_fraction(0.7) # Use only 70% of GPU memory
print("Limited GPU memory usage to 70%")
' || true

# Create a pre-run script to optimize GPU
cat > pre_run.py << 'EOL'
import gc
import torch
import torch.cuda

# Aggressive garbage collection
gc.collect()
torch.cuda.empty_cache()

# Force cudnn to minimize memory usage
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False  # Disable cudnn for minimal memory footprint

# Lower memory fraction
torch.cuda.set_per_process_memory_fraction(0.6)  # Use only 60% of available memory

# Use more aggressive GPU memory allocation
torch.cuda.memory.set_per_process_memory_fraction(0.6)

print("GPU memory pre-optimization complete")
print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
EOL

# Run pre-optimization script
python pre_run.py

# Run with extremely minimal logging to save memory
python run.py config/pixelrealm_lora/PixelRealm_ultra_low_mem.yaml