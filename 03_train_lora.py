"""
Workshop Step 3: Parameter-Efficient Fine-Tuning (PEFT / LoRA)
---------------------------------------------------------------
This module trains low-rank adaptation (LoRA) adapter matrices on top of 
the base model using Hugging Face TRL SFTTrainer.
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

BASE_MODEL_DIR = "./LFM2.5-230M"
DATASET_PATH = "train_formatted.jsonl"
LORA_OUTPUT_DIR = "./essay_reviewer_lora"

def train_lora(
    base_model_dir=BASE_MODEL_DIR,
    dataset_path=DATASET_PATH,
    lora_output_dir=LORA_OUTPUT_DIR,
    epochs=3,
    lr=2e-4
):
    import torch

    print(f"[Step 3] Loading base model from '{base_model_dir}'...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    print(f"[Step 3] Loading formatted dataset from '{dataset_path}'...")
    dataset = load_dataset("json", data_files=dataset_path)

    print("[Step 3] Configuring LoRA rank adapter (r=8, alpha=16)...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        packing=False,
        dataset_text_field="text",
        max_length=128,
        bf16=False,
        use_cpu=True,
        dataloader_pin_memory=False,
    )

    print("[Step 3] Initializing SFTTrainer and launching fine-tuning...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        peft_config=lora_config,
        args=training_args,
    )

    trainer.train()

    print(f"[Step 3] Fine-tuning complete. Saving LoRA adapter to '{lora_output_dir}'...")
    trainer.save_model(lora_output_dir)
    print("[Step 3] LoRA adapter weights saved successfully!")
    return lora_output_dir

if __name__ == "__main__":
    train_lora()
