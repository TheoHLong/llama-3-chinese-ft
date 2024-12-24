from transformers import TextStreamer
from unsloth import FastLanguageModel
import subprocess
import time

def setup_model(model_path, max_seq_length=2048, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def format_prompt(instruction, input_text="", output=""):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""

def generate_response(model, tokenizer, instruction, input_text="", max_new_tokens=128, stream=True):
    prompt = format_prompt(instruction, input_text)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    if stream:
        text_streamer = TextStreamer(tokenizer)
        outputs = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
        return tokenizer.batch_decode(outputs)

def save_gguf(model, output_path, quant_method="q8_0"):
    model.save_pretrained_gguf(output_path, quant_method=quant_method)

def save_model_formats(model, base_path):
    # Save to multiple GGUF formats
    model.push_to_hub_gguf(
        base_path,
        quantization_method=["q4_k_m", "q8_0", "q5_k_m", "f16"],
    )

def setup_ollama():
    subprocess.Popen(["ollama", "serve"])
    time.sleep(3)  # Wait for Ollama to start

def export_to_ollama(model_name, model, tokenizer):
    # First save as GGUF
    model.save_pretrained_gguf(f"{model_name}_gguf")
    
    # Create Ollama model
    subprocess.run(["ollama", "create", model_name, "-f", "./model/Modelfile"])

if __name__ == "__main__":
    # Example usage
    model, tokenizer = setup_model("output/final_model")
    
    print("Testing English generation:")
    generate_response(
        model,
        tokenizer,
        instruction="Continue the Fibonacci sequence",
        input_text="1, 1, 2, 3, 5, 8",
    )

    print("\nTesting Chinese generation:")
    generate_response(
        model,
        tokenizer,
        instruction="人为什么需要工作？",
    )