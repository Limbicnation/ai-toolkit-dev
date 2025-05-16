#!/bin/bash

# Create a backup directory for safety
BACKUP_DIR="/home/gero/libstdc_backup"
mkdir -p $BACKUP_DIR

# Path to your conda environment
CONDA_ENV="/home/gero/anaconda3/envs/ai-toolkit"
CONDA_LIB="$CONDA_ENV/lib"

echo "Backing up original library..."
# Backup the original file
cp "$CONDA_LIB/libstdc++.so.6" "$BACKUP_DIR/libstdc++.so.6.backup"

echo "Creating symbolic link to system libstdc++..."
# Remove existing symlink/file in conda environment
rm -f "$CONDA_LIB/libstdc++.so.6"

# Create a new symlink to system libstdc++
ln -s /lib/x86_64-linux-gnu/libstdc++.so.6 "$CONDA_LIB/libstdc++.so.6"

echo "Verification:"
strings "$CONDA_LIB/libstdc++.so.6" | grep GLIBCXX_3.4.32

echo "Done! Your conda environment now uses the system libstdc++ which has GLIBCXX_3.4.32."
echo "To revert this change, run: mv $BACKUP_DIR/libstdc++.so.6.backup $CONDA_LIB/libstdc++.so.6"