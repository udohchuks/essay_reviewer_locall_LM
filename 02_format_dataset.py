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

    print(f"[Step 2] Formatting dataset...")
    
    import random

    PROMPT_VARIATIONS = [
        "Review this essay and give feedback:",
        "Please critique the following essay and suggest improvements:",
        "Analyze this essay and provide constructive evaluation:",
        "Read this student essay and offer detailed feedback:",
        "Evaluate the clarity, structure, and content of this essay:"
    ]

    use_tokenizer = os.path.exists(base_model_dir)
    tokenizer = None
    if use_tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
            print("[Step 2] Loaded tokenizer from local model directory.")
        except Exception:
            use_tokenizer = False

    def apply_template(example):
        prefix = random.choice(PROMPT_VARIATIONS)
        user_content = f"{prefix}\n\n{example['essay']}"
        assistant_content = example['feedback']

        if use_tokenizer and tokenizer:
            messages = [
                {"role": "user", "content": user_content},  
                {"role": "assistant", "content": assistant_content}
            ]
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
        else:
            # ChatML Standard Format Fallback
            formatted = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}<|im_end|>\n"
            return {"text": formatted}

    print("[Step 2] Applying Chat Template to dataset entries...")
    formatted_dataset = dataset.map(apply_template)
    formatted_dataset = formatted_dataset.remove_columns(["essay", "feedback"])
    
    formatted_dataset["train"].to_json(output_path)
    print(f"[Step 2] Formatted dataset successfully saved to '{output_path}'!")
    return output_path

if __name__ == "__main__":
    format_dataset()
