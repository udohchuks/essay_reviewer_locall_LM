# 📝 Essay Reviewer Local Language Model Workshop

Welcome to the **Hands-On Workshop on Building, Fine-Tuning, & Deploying Small Language Models Locally**!

In this workshop, you will build a domain-specific **Essay Feedback Assistant** using an ultra-compact 230M parameter foundation model (`LiquidAI/LFM2.5-230M`). You will learn how to fine-tune it with **PEFT/LoRA**, merge model weights, convert checkpoints into **GGUF format**, and run ultra-fast CPU inference using pre-built **`llama.cpp`** binaries.

---

## 🎯 Workshop Objectives

By the end of this session, participants will understand and implement:
1. **The End-to-End SLM Lifecycle**: Data Prep $\rightarrow$ LoRA Fine-Tuning $\rightarrow$ Weight Merging $\rightarrow$ GGUF Quantization $\rightarrow$ Local Inference.
2. **Dataset Formatting & Chat Templates**: Converting raw text pairs into standardized prompt formats using Hugging Face tokenizers.
3. **Parameter-Efficient Fine-Tuning (PEFT / LoRA)**: Training low-rank adapters (`r=8, alpha=16`) on target linear layers using `SFTTrainer`.
4. **Model Weight Merging**: Fusing adapter matrices back into the base model parameters.
5. **GGUF Conversion & Quantization**: Converting PyTorch models into GGUF (`f16` and 8-bit `Q8_0`) for CPU execution.
6. **Local OpenAI-Compatible Server**: Running low-latency, private LLM web services via `llama-server`.

---

## 🏗️ Repository Architecture

```
essay_reviewer_locall_LM/
├── train.jsonl                 # Original raw essay feedback dataset
├── train_formatted.jsonl       # Formatted chat template dataset
├── requirements.txt            # Python dependencies
├── setup_llamacpp.py           # Prebuilt llama.cpp binary downloader & extractor
│
├── 01_download_base_model.py   # Step 1: Download foundation model (LFM2.5-230M)
├── 02_format_dataset.py        # Step 2: Format dataset with Chat Template
├── 03_train_lora.py            # Step 3: Train LoRA adapter with SFTTrainer
├── 04_test_python.py           # Step 4: Test fine-tuned adapter in Python
├── 05_merge_model.py           # Step 5: Merge LoRA weights into base model
├── 06_convert_and_quantize.py  # Step 6: Convert to GGUF (f16) & quantize (Q8_0)
├── convert_hf_to_gguf.py       # Custom GGUF model converter script
├── conversion/                 # LFM2 GGUF conversion module
│   ├── __init__.py
│   ├── base.py
│   └── lfm2.py                 # Liquid AI LFM2 conversion rules
│
├── LFM2.5-230M/                # Local base model cache (Git ignored)
├── essay_reviewer_lora/        # Saved LoRA adapter weights (Git ignored)
├── essay_reviewer_merged/      # Fully merged HF model (Git ignored)
├── essay_reviewer_finetuned_q8.gguf # Quantized GGUF local model (Git ignored)
└── README.md                   # Complete workshop guide
```

---

## 🚀 Environment & Setup

### 1. Create Virtual Environment & Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Fast installation using uv (recommended):
uv pip install -r requirements.txt
# Standard pip fallback:
pip install -r requirements.txt
```

### 2. Download Prebuilt `llama.cpp` Binaries (Step 0)

Instead of compiling C++ code from source, run the binary setup helper:

```bash
python setup_llamacpp.py
```

**Code Explanation**:
```python
# setup_llamacpp.py queries GitHub API for ggml-org/llama.cpp prebuilt releases
# and extracts binaries into ./llama.cpp_bin/
url, filename = get_latest_release_url()
download_and_extract(url=url)
```
*Creates `./llama.cpp_bin/` containing `llama-server.exe`, `llama-quantize.exe`, and `llama-cli.exe`.*

---

## 📚 Step-by-Step Workshop Guide & Code Walkthrough

---

### Step 1: Download Base Foundation Model (`01_download_base_model.py`)

**Command**:
```bash
python 01_download_base_model.py
```

**Code Walkthrough**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2.5-230M"
LOCAL_DIR = "./LFM2.5-230M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(LOCAL_DIR)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.save_pretrained(LOCAL_DIR)
```

**Explanation**:
- **Why LiquidAI/LFM2.5-230M?**: It is an ultra-compact 230M parameter hybrid model. At only ~460MB in FP16, it can be fine-tuned and run directly on laptop CPUs without needing expensive GPU cloud hardware.
- Downloads model weights and tokenizer configurations locally to `./LFM2.5-230M`.

---

### Step 2: Prepare & Format Dataset (`02_format_dataset.py`)

**Command**:
```bash
python 02_format_dataset.py
```

**Code Walkthrough**:
```python
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./LFM2.5-230M")

with open("train.jsonl", "r") as infile, open("train_formatted.jsonl", "w") as outfile:
    for line in infile:
        data = json.loads(line)
        messages = [
            {"role": "user", "content": f"Review this essay and provide constructive feedback:\n\n{data['essay']}"},
            {"role": "assistant", "content": data["feedback"]}
        ]
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
        outfile.write(json.dumps({"text": formatted_text}) + "\n")
```

**Explanation**:
- **Why Chat Templates?**: Raw text must be structured into role-delimited turns (`user`, `assistant`) so the LLM clearly distinguishes user queries from target responses.
- `tokenizer.apply_chat_template(messages, tokenize=False)` injects special control tokens (e.g. `<|im_start|>user`, `<|im_end|>`) specific to the model.

---

### Step 3: Parameter-Efficient Fine-Tuning with LoRA (`03_train_lora.py`)

**Command**:
```bash
python 03_train_lora.py
```

**Code Walkthrough**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("./LFM2.5-230M")
tokenizer = AutoTokenizer.from_pretrained("./LFM2.5-230M")
dataset = load_dataset("json", data_files="train_formatted.jsonl")

# Configure LoRA adapter
lora_config = LoraConfig(
    r=8,                       # Rank dimension of adapter matrices
    lora_alpha=16,             # Scaling factor (alpha / r = 2.0)
    target_modules=["q_proj", "v_proj"], # Inject into query and value projections
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    dataset_text_field="text",
    max_length=128,            # Short length for fast CPU demonstration
    use_cpu=True,
    dataloader_pin_memory=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    peft_config=lora_config,
    args=training_args,
)

trainer.train()
trainer.save_model("./essay_reviewer_lora")
```

**Explanation**:
- **Why LoRA (Low-Rank Adaptation)?**: Instead of updating all 230M parameters (expensive), LoRA freezes the original model and injects small pair matrices $W = W_0 + \Delta W$ where $\Delta W = B \times A$.
- **Parameters**: `r=8` sets low-rank matrix rank; `target_modules=["q_proj", "v_proj"]` targets key attention weights.
- Saves adapter weights (~2MB) to `./essay_reviewer_lora`.

---

### Step 4: Python Inference & Evaluation (`04_test_python.py`)

**Command**:
```bash
python 04_test_python.py
```

**Code Walkthrough**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("./LFM2.5-230M")
base_model = AutoModelForCausalLM.from_pretrained("./LFM2.5-230M")

# Attach fine-tuned LoRA adapter to base model
model = PeftModel.from_pretrained(base_model, "./essay_reviewer_lora")

message = [{"role": "user", "content": "Review this essay:\n\nStandardized testing provides uniform metrics..."}]
prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=256,
    do_sample=False,
)

feedback = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(feedback)
```

**Explanation**:
- Dynamically loads base model + adapter weights in Python and generates structured essay feedback to qualitatively evaluate fine-tuning results.

---

### Step 5: Merge LoRA Adapter Weights into Base Model (`05_merge_model.py`)

**Command**:
```bash
python 05_merge_model.py
```

**Code Walkthrough**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("./LFM2.5-230M")
model = PeftModel.from_pretrained(base_model, "./essay_reviewer_lora")

# Permanently fuse adapter matrices into base model weights
merged_model = model.merge_and_unload()

merged_model.save_pretrained("./essay_reviewer_merged")
tokenizer = AutoTokenizer.from_pretrained("./LFM2.5-230M")
tokenizer.save_pretrained("./essay_reviewer_merged")
```

**Explanation**:
- **Why Merge?**: C++ inference engines like `llama.cpp` load single standalone model files. `merge_and_unload()` computes $W_{merged} = W_0 + (B \times A)$ and saves a complete standalone Hugging Face model folder `./essay_reviewer_merged`.

---

### Step 6: Convert to GGUF & Quantize (`06_convert_and_quantize.py`)

**Command**:
```bash
python 06_convert_and_quantize.py
```

**Code Walkthrough**:
```python
import subprocess

# Phase 1: Convert HF merged model to GGUF F16 format
cmd_convert = [
    ".venv/Scripts/python.exe", "convert_hf_to_gguf.py",
    "./essay_reviewer_merged",
    "--outfile", "essay_reviewer_finetuned_f16.gguf",
    "--outtype", "f16"
]
subprocess.run(cmd_convert, check=True)

# Phase 2: Quantize F16 GGUF down to 8-bit Q8_0
cmd_quantize = [
    "./llama.cpp_bin/llama-quantize.exe",
    "essay_reviewer_finetuned_f16.gguf",
    "essay_reviewer_finetuned_q8.gguf",
    "Q8_0"
]
subprocess.run(cmd_quantize, check=True)
```

**Explanation**:
- **Why GGUF & Conversion?**: PyTorch `.safetensors` files require Python runtime overhead. `convert_hf_to_gguf.py` converts Hugging Face tensors into GGUF binary format optimized for CPU SIMD/AVX instructions.
- **Why Quantization?**: `Q8_0` compresses 16-bit floating point parameters down to 8-bit integers. This reduces model size from ~460MB to ~232MB and doubles execution speed with virtually zero loss in quality.

---

### Step 7: Launch Local OpenAI-Compatible API Server

**Command**:
```bash
# Windows
.\llama.cpp_bin\llama-server.exe -m essay_reviewer_finetuned_q8.gguf --port 8080

# macOS / Linux
./llama.cpp_bin/llama-server -m essay_reviewer_finetuned_q8.gguf --port 8080
```

**Explanation**:
- Launches `llama-server.exe`, an ultra-fast C++ web server loading `essay_reviewer_finetuned_q8.gguf`.
- Provides an **OpenAI-compatible REST API** on `http://127.0.0.1:8080`.

**Testing the Local Server (cURL)**:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Review this essay: Standardized testing provides uniform metrics but can limit evaluation."}
    ]
  }'
```

---

## 🛠️ Summary Command Pipeline

Execute all steps sequentially:

```bash
python setup_llamacpp.py
python 01_download_base_model.py
python 02_format_dataset.py
python 03_train_lora.py
python 04_test_python.py
python 05_merge_model.py
python 06_convert_and_quantize.py
.\llama.cpp_bin\llama-server.exe -m essay_reviewer_finetuned_q8.gguf --port 8080
```

---

## 💡 Troubleshooting & FAQs

- **Why did GGUF conversion need custom code (`conversion/lfm2.py`)?**:
  Standard architectures (Llama, Mistral) have default conversion rules. Liquid AI LFM2.5 is a new hybrid architecture (`Lfm2ForCausalLM`) requiring custom layer tensor mappings supplied in `conversion/lfm2.py`.
- **How to speed up training during workshops**:
  Set `epochs=1` or `max_length=128` in `03_train_lora.py`.
- **Pre-workshop recommendation**:
  Pre-install `.venv` and pre-download `./LFM2.5-230M` before the workshop so participants don't rely on heavy venue Wi-Fi downloads.

---

*Happy Fine-Tuning!* 🚀
