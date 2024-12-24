# Usage Guide

## Training

1. Start Training
   ```bash
   bash scripts/run_training.sh
   ```

2. Monitor Training
   - View progress in terminal
   - Check WandB dashboard (if configured)
   - Monitor GPU usage with `nvidia-smi`

3. Training Output
   - Models saved in `output/final_model/`
   - Logs saved in `logs/`

## Inference

1. Run Inference
   ```bash
   bash scripts/run_inference.sh [model_path]
   ```

2. Inference Options
   - Stream generation
   - Batch processing
   - Temperature and top-p sampling

## Model Export

1. Export to GGUF
   ```python
   from src.inference import ModelInference
   
   model = ModelInference("path/to/model")
   model.export_to_gguf("output_path", quant_method="q4_k_m")
   ```

2. Quantization Options
   - q8_0: Fast conversion, higher resource use
   - q4_k_m: Recommended balanced option
   - q5_k_m: Higher quality, larger size

## Ollama Integration

1. Export to Ollama
   ```python
   model.export_to_ollama("model_name")
   ```

2. Using with Ollama
   ```bash
   ollama run model_name
   ```

## Common Issues

1. Out of Memory
   - Reduce batch size
   - Use gradient accumulation
   - Enable 4-bit quantization

2. Training Issues
   - Check learning rate
   - Verify data format
   - Monitor loss values