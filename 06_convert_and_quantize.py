"""
Workshop Step 6: GGUF Conversion & Quantization
------------------------------------------------
This module converts the merged Hugging Face model checkpoint into 
GGUF format (f16) and quantizes it to high-efficiency 8-bit (q8_0) or 4-bit (q4_k_m).
"""

import os
import sys
import subprocess

MERGED_MODEL_DIR = "./essay_reviewer_merged"
F16_GGUF = "essay_reviewer_finetuned_f16.gguf"
Q8_GGUF = "essay_reviewer_finetuned_q8.gguf"

def find_convert_script():
    """Locate convert_hf_to_gguf.py in the environment."""
    candidates = [
        "convert_hf_to_gguf.py",
        os.path.join("llama.cpp", "convert_hf_to_gguf.py"),
        os.path.join("llama.cpp_bin", "convert_hf_to_gguf.py"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None

def find_quantize_binary():
    """Locate llama-quantize executable."""
    candidates = [
        "llama-quantize",
        "llama-quantize.exe",
        os.path.join("llama.cpp_bin", "llama-quantize.exe"),
        os.path.join("llama.cpp_bin", "llama-quantize"),
        os.path.join("llama.cpp", "llama-quantize.exe"),
        os.path.join("llama.cpp", "llama-quantize"),
        os.path.join("llama.cpp", "build", "bin", "llama-quantize"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return "llama-quantize"

def convert_to_gguf(merged_dir=MERGED_MODEL_DIR, output_f16=F16_GGUF):
    script = find_convert_script()
    if not script:
        print("[Error] Could not find 'convert_hf_to_gguf.py'. Make sure it is in your project directory.")
        return False
    
    cmd = [sys.executable, script, merged_dir, "--outfile", output_f16, "--outtype", "f16"]
    print(f"[Step 6] Converting HF model to F16 GGUF: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def quantize_gguf(input_f16=F16_GGUF, output_q8=Q8_GGUF, quant_type="Q8_0"):
    quantize_bin = find_quantize_binary()
    cmd = [quantize_bin, input_f16, output_q8, quant_type]
    print(f"[Step 6] Quantizing model ({quant_type}): {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except FileNotFoundError:
        print(f"[Error] '{quantize_bin}' binary not found. Run setup_llamacpp.py or install prebuilt binaries.")
        return False

def main():
    print("=== GGUF Conversion & Quantization Pipeline ===")
    if not os.path.exists(MERGED_MODEL_DIR):
        print(f"[Error] Merged model directory '{MERGED_MODEL_DIR}' does not exist. Run Step 5 first.")
        return

    # Step 6a: Convert to F16 GGUF
    print("\n-- Phase 1: HF to GGUF (f16) --")
    success = convert_to_gguf()
    if not success:
        print("Conversion failed.")
        return

    # Step 6b: Quantize GGUF to Q8_0
    print("\n-- Phase 2: GGUF Quantization (Q8_0) --")
    success = quantize_gguf()
    if success:
        print(f"\n[Step 6 Complete] Model successfully quantized and ready in '{Q8_GGUF}'!")
    else:
        print("Quantization failed.")

if __name__ == "__main__":
    main()
