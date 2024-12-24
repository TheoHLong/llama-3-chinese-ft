import psutil
from datasets import load_dataset
from torch.utils.data import DataLoader

def format_alpaca_prompt(instruction, input_text, output):
    """Format the prompt in Alpaca style."""
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""

def formatting_prompts_func(examples, tokenizer):
    en_instructions = examples["en_instruction"]
    en_inputs = examples["en_input"]
    en_outputs = examples["en_output"]

    zh_instructions = examples["zh_instruction"]
    zh_inputs = examples["zh_input"]
    zh_outputs = examples["zh_output"]

    all_texts = []
    for en_inst, en_inp, en_out, zh_inst, zh_inp, zh_out in zip(
        en_instructions, en_inputs, en_outputs,
        zh_instructions, zh_inputs, zh_outputs
    ):
        en_text = format_alpaca_prompt(en_inst, en_inp, en_out) + tokenizer.eos_token
        zh_text = format_alpaca_prompt(zh_inst, zh_inp, zh_out) + tokenizer.eos_token
        all_texts.append(en_text)
        all_texts.append(zh_text)

    return {"text": all_texts}

def prepare_dataset(dataset_path, tokenizer, max_length, batch_size):
    # Get CPU count
    num_cores = psutil.cpu_count(logical=True)
    num_physical_cores = psutil.cpu_count(logical=False)
    
    print(f"Total CPU cores (logical): {num_cores}")
    print(f"Physical CPU cores: {num_physical_cores}")

    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Apply text formatting
    dataset = dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_physical_cores
    )

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_physical_cores,
        remove_columns=["text"]
    )

    # Set format to PyTorch tensors
    tokenized_dataset.set_format("torch")

    return tokenized_dataset