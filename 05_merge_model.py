"""
Workshop Step 5: Bake LoRA Weights into Base Model (Merge)
----------------------------------------------------------
This module merges the trained LoRA adapter weights directly into 
the base model's weight matrices and exports a complete Hugging Face checkpoint.
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_DIR = "./LFM2.5-230M"
LORA_DIR = "./essay_reviewer_lora"
MERGED_OUTPUT_DIR = "./essay_reviewer_merged"

def merge_and_save(
    base_model_dir=BASE_MODEL_DIR,
    lora_dir=LORA_DIR,
    output_dir=MERGED_OUTPUT_DIR
):
    print(f"[Step 5] Loading base model from '{base_model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_dir)

    print(f"[Step 5] Applying LoRA adapter from '{lora_dir}'...")
    model = PeftModel.from_pretrained(base_model, lora_dir)

    print("[Step 5] Merging adapter weights into base model matrices...")
    merged_model = model.merge_and_unload()

    print(f"[Step 5] Saving merged model checkpoint to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("[Step 5] Model merge complete! Ready for GGUF conversion.")
    return output_dir

if __name__ == "__main__":
    merge_and_save()
