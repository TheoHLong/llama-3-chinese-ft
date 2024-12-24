#!/bin/bash

# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY is not set. Training will proceed without logging to W&B."
fi

# Create output directory if it doesn't exist
mkdir -p output

# Run training script
echo "Starting training..."
python src/model_training.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
    exit 1
fi