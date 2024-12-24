#!/bin/bash

# Default values
MODEL_PATH="output/final_model"
EXPORT_TYPE="all"  # Can be: gguf, ollama, or all
QUANT_METHOD="q4_k_m"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --export-type)
            EXPORT_TYPE="$2"
            shift 2
            ;;
        --quant-method)
            QUANT_METHOD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create Python script for export
cat << EOF > export_script.py
from src.inference import ModelInference
import sys

model = ModelInference("${MODEL_PATH}")

if "${EXPORT_TYPE}" in ["gguf", "all"]:
    # Export to GGUF
    if "${QUANT_METHOD}" == "all":
        model.export_multiple_gguf("${MODEL_PATH}_gguf")
    else:
        model.export_gguf("${MODEL_PATH}_gguf", "${QUANT_METHOD}")
    print("GGUF export completed")

if "${EXPORT_TYPE}" in ["ollama", "all"]:
    # Export to Ollama
    model.setup_ollama()
    model.export_to_ollama("unsloth_model")
    print("Ollama export completed")
EOF

# Run the export script
python export_script.py

# Cleanup
rm export_script.py

echo "Export completed successfully!"