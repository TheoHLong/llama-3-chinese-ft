# Setup Guide

## Environment Setup

1. System Requirements
   - CUDA-capable GPU with at least 16GB VRAM
   - Python 3.8+
   - Git

2. Installation
   ```bash
   # Clone the repository
   git clone [repository-url]
   cd llama-3-chinese-ft

   # Run setup script
   bash scripts/setup_env.sh
   ```

3. Environment Variables
   ```bash
   export WANDB_API_KEY=your_wandb_key  # Optional, for logging
   ```

## Data Preparation

1. Dataset Format
   The project expects data in JSON format with the following structure:
   ```json
   {
     "en_instruction": "...",
     "en_input": "...",
     "en_output": "...",
     "zh_instruction": "...",
     "zh_input": "...",
     "zh_output": "..."
   }
   ```

2. Data Configuration
   Update the dataset path in `configs/training_config.yaml`:
   ```yaml
   data:
     dataset_path: "path/to/your/dataset.json"
     num_proc: 4
   ```

## Configuration

1. Model Configuration
   - Adjust model parameters in `configs/training_config.yaml`
   - Configure LoRA parameters
   - Set batch size and learning rate

2. Training Configuration
   - Set number of epochs
   - Configure gradient accumulation
   - Set warmup steps
   - Choose optimizer settings