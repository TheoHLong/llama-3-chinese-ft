# Chinese Fine-tuning for Llama 3.1 8B

This repository implements fine-tuning of the Llama 3.1 8B model on Chinese language data using the Unsloth framework for 2x faster training.

## Features

- Fine-tunes Llama 3.1 8B on Chinese-English bilingual dataset
- Uses LoRA adapters for efficient training
- 2x faster training with Unsloth optimization
- Support for multiple export formats (GGUF, float16, 4-bit)
- Ollama integration for easy deployment

## Installation

```bash
pip install -r requirements.txt
```

## Training

1. Login to Weights & Biases:
```python
import wandb
wandb.login()
```

2. Configure your parameters in `configs/training_config.yaml`

3. Run training:
```bash
python src/model_training.py
```

## Model Export

### Export to GGUF
```python
from src.inference import ModelInference

model = ModelInference("output/final_model")
# Single format
model.export_gguf("output_path", quant_method="q4_k_m")
# Multiple formats
model.export_multiple_gguf("output_path", ["q4_k_m", "q8_0", "q5_k_m", "f16"])
```

### Export to Ollama
```python
model.setup_ollama()
model.export_to_ollama("model_name")
```

## Example Usage

```python
from src.inference import ModelInference

model = ModelInference("output/final_model")

# English Example
model.generate(
    instruction="Continue the Fibonacci sequence",
    input_text="1, 1, 2, 3, 5, 8",
    stream=True
)

# Chinese Example
model.generate(
    instruction="人为什么需要工作？",
    stream=True
)
```

## Project Structure
```
llama-3-chinese-ft/
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── src/
│   ├── data_preparation.py   # Dataset processing
│   ├── model_training.py     # Training implementation
│   └── inference.py          # Inference and exports
├── configs/
│   └── training_config.yaml  # Training parameters
└── scripts/
    └── export_model.sh       # Model export utilities
```

## Performance

- 2x faster training with Unsloth optimization
- Memory-efficient with 4-bit quantization
- Supports various GGUF export options for deployment
