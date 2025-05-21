#!/bin/bash

# Update bitsandbytes to the version that works with newer PyTorch
pip install --upgrade bitsandbytes==0.45.5

# Make sure we have the right triton version
pip install triton==3.2.0

# Update or reinstall diffusers to get the required components
pip install --upgrade diffusers

# Output message
echo "Dependencies updated. Please try running your command again."