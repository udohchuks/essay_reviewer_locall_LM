"""
Workshop Step 1: Download Base Model & Tokenizer
------------------------------------------------
This module downloads the lightweight foundation LLM (LiquidAI/LFM2.5-230M)
and saves it locally to avoid repeated internet downloads during fine-tuning.
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2.5-230M"
OUTPUT_DIR = "./LFM2.5-230M"

def download_base_model(model_id=MODEL_ID, output_dir=OUTPUT_DIR):
    print(f"[Step 1] Fetching base model '{model_id}' from Hugging Face Hub...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    os.makedirs(output_dir, exist_ok=True)
    print(f"[Step 1] Saving model and tokenizer locally to '{output_dir}'...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("[Step 1] Base model and tokenizer successfully saved!")
    return output_dir

if __name__ == "__main__":
    download_base_model()
