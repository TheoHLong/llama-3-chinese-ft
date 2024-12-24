#!/bin/bash

# Check if model path is provided
MODEL_PATH=${1:-"output/final_model"}
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path $MODEL_PATH does not exist!"
    exit 1
fi

# Run inference script
echo "Starting inference..."
python src/inference.py --model_path $MODEL_PATH

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo "Inference completed successfully!"
else
    echo "Inference failed with exit code $?"
    exit 1
fi