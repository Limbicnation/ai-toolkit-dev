#!/bin/bash

# This script runs the PixelRealm LoRA training with memory optimizations

# Set PyTorch memory optimization variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow messages
export HF_HOME=/home/gero/.cache/huggingface  # Use system HuggingFace cache
export TRANSFORMERS_CACHE=/home/gero/.cache/huggingface/hub
export HF_HUB_CACHE=/home/gero/.cache/huggingface/hub

# HuggingFace authentication - add your token here
# Get your token from: https://huggingface.co/settings/tokens
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. You may need to authenticate for gated models."
    echo "Set your token with: export HF_TOKEN=your_token_here"
    echo "Get your token from: https://huggingface.co/settings/tokens"
fi
export HF_TOKEN

# Check if dataset is empty and copy sample data if needed
DATASET_DIR="./dataset/PixelRealm_LoRA-Dataset_v5_2025-05-30"
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

# Function to verify and download FLUX model
verify_and_download_model() {
    echo "Verifying FLUX model availability..."
    local model_name="black-forest-labs/FLUX.1-dev"
    local expected_files=(
        "pytorch_model-00001-of-00003.safetensors"
        "pytorch_model-00002-of-00003.safetensors" 
        "pytorch_model-00003-of-00003.safetensors"
    )
    
    # Check if model files exist and are complete
    local model_path="$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev"
    local missing_files=0
    
    for file in "${expected_files[@]}"; do
        local file_path=$(find "$model_path" -name "$file" 2>/dev/null | head -1)
        if [ ! -f "$file_path" ]; then
            echo "Missing or incomplete: $file"
            missing_files=$((missing_files + 1))
        else
            local file_size=$(stat -c%s "$file_path" 2>/dev/null || echo 0)
            if [ "$file_size" -lt 1000000000 ]; then  # Less than 1GB suggests incomplete
                echo "File appears incomplete: $file (size: $file_size bytes)"
                missing_files=$((missing_files + 1))
            fi
        fi
    done
    
    if [ "$missing_files" -gt 0 ]; then
        echo "Model files missing or incomplete. Attempting download..."
        python -c "
import os
from huggingface_hub import snapshot_download
import sys

try:
    print('Downloading FLUX model with retry logic...')
    snapshot_download(
        repo_id='$model_name',
        cache_dir='$HF_HOME/hub',
        local_files_only=False,
        token=os.environ.get('HF_TOKEN'),
        max_workers=4
    )
    print('Model download completed successfully!')
except Exception as e:
    print(f'Download failed: {e}')
    sys.exit(1)
"
        if [ $? -ne 0 ]; then
            echo "Model download failed. Please check your HF_TOKEN and internet connection."
            exit 1
        fi
    else
        echo "All model files verified successfully!"
    fi
}

# Verify and download model
verify_and_download_model 

# Run the training with low memory settings
echo "Starting training with memory-optimized settings..."
python run.py config/pixelchar_refined_v8.yaml -l ./output/training.log