"""
Workshop Step 2: Prepare & Format Fine-Tuning Dataset
------------------------------------------------------
This module formats raw user essay & feedback pairs into chat template strings 
compatible with Hugging Face SFTTrainer.
"""

import os
from transformers import AutoTokenizer
from datasets import load_dataset

INPUT_JSONL = "train.jsonl"
OUTPUT_JSONL = "train_formatted.jsonl"
BASE_MODEL_DIR = "./LFM2.5-230M"

def format_dataset(input_path=INPUT_JSONL, output_path=OUTPUT_JSONL, base_model_dir=BASE_MODEL_DIR):
    print(f"[Step 2] Loading raw dataset from '{input_path}'...")
    dataset = load_dataset("json", data_files=input_path)

    print(f"[Step 2] Loading tokenizer from '{base_model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    def apply_template(example):
        messages = [
            {"role": "user", "content": f"Review this essay and give feedback:\n\n{example['essay']}"},  
            {"role": "assistant", "content": f"{example['feedback']}"}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    print("[Step 2] Applying Chat Template to dataset entries...")
    formatted_dataset = dataset.map(apply_template)
    formatted_dataset = formatted_dataset.remove_columns(["essay", "feedback"])
    
    formatted_dataset["train"].to_json(output_path)
    print(f"[Step 2] Formatted dataset successfully saved to '{output_path}'!")
    return output_path

if __name__ == "__main__":
    format_dataset()
