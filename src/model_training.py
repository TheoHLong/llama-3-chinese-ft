import os
import torch
import wandb
import yaml
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from data_preparation import prepare_dataset

# Initialize wandb - use environment variable WANDB_API_KEY
wandb.login()

def print_gpu_memory_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return start_gpu_memory, max_memory

def train_model(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(
        project="Llama_3.1_8b_Chinese_FT",
        config={
            "learning_rate": config['training']['learning_rate'],
            "architecture": "Llama_3.1_8b",
            "dataset": "alpaca-chinese-52k-v3",
            "epochs": config['training']['num_train_epochs'],
            "optim": "adamw_8bit",
            "weight_decay": config['training']['weight_decay'],
            "lr_scheduler_type": "linear",
            "seed": 3407,
            "per_device_train_batch_size": 64,
            "gradient_accumulation_steps": 32,
            "warmup_steps": 3,
        }
    )

    # Print initial GPU stats
    start_gpu_memory, max_memory = print_gpu_memory_stats()

    # Initialize model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B",
        max_seq_length=2048,
        load_in_4bit=True
    )

    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    # Prepare dataset
    dataset = prepare_dataset(
        config['data']['dataset_path'],
        tokenizer,
        2048,
        64
    )

    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="input_ids",
        max_seq_length=2048,
        dataset_num_proc=4,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=64,
            gradient_accumulation_steps=32,
            warmup_steps=3,
            num_train_epochs=1,
            learning_rate=1e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="output",
            report_to="wandb",
        )
    )

    # Train
    trainer_stats = trainer.train()

    # Print final memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory/max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save model
    model.save_pretrained("output/final_model")
    tokenizer.save_pretrained("output/final_model")

if __name__ == "__main__":
    train_model("configs/training_config.yaml")