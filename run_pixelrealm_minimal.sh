#!/bin/bash

# This script runs an extremely minimized version of PixelRealm LoRA training
# Use this if you're still encountering memory issues

# Set PyTorch memory optimization variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow messages

# Create a temporary minimal configuration file
cat > ./config/pixelrealm_lora/PixelRealm_minimal.yaml << 'EOL'
job: extension
config:
  name: "PixelRealm_LoRA_Minimal"
  process:
    - type: 'sd_trainer'
      training_folder: "output/pixelrealm_lora_minimal"
      device: cuda:0
      trigger_word: "pixelrealm"
      
      network:
        type: "lora"
        linear: 8
        linear_alpha: 8
        conv: 4
        conv_alpha: 4
        dropout: 0.1
      
      save:
        dtype: float16
        save_every: 1000
        max_step_saves_to_keep: 3
        push_to_hub: false
        safe_serialization: true
      
      datasets:
        - folder_path: "./dataset/PixelRealm_LoRA-Dataset_v1_2025-05-12"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: true
          cache_latents_to_disk: true
          resolution: [256, 256]
      
      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 4
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        lr_scheduler: "constant"
        skip_first_sample: true
        low_vram: true
        mixed_precision: "fp16"
        gradient_clip_norm: 1.0
        max_grad_norm: 1.0
        use_8bit_adam: true
        
        ema_config:
          use_ema: false
        
        dtype: "fp16"
      
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true
        qtype: "qint4"
        low_vram: true
        revision: "main"
        local_files_only: false
        use_safetensors: true
        enable_xformers: true
        model_cpu_offload: true
        attention_mode: "xformers"
        max_memory: {0: "6GB"}
        device_map: "balanced_low_0"
        split_model_over_gpus: false
        offload_to_cpu: true
      
      sample:
        sampler: "flowmatch"
        sample_every: 2000
        width: 256
        height: 256
        prompts:
          - "[trigger] pixel character, simple style"
        neg: "blurry, low quality"
        seed: 42
        guidance_scale: 5
        sample_steps: 15

meta:
  name: "PixelRealm LoRA Minimal"
  version: '1.0'
EOL

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
pip install accelerate==0.28.0

# Run the minimal training
echo "Starting training with absolute minimal settings..."
python run.py config/pixelrealm_lora/PixelRealm_minimal.yaml