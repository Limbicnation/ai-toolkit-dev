#!/bin/bash

# Update bitsandbytes to the version that works with newer PyTorch
pip install --upgrade bitsandbytes==0.45.5

# Make sure we have the right triton version
pip install triton==3.2.0

# Run the configuration file
python run.py config/pixelrealm_lora/PixelRealm_LoRA-Dataset_v1_2025-05-12.yaml