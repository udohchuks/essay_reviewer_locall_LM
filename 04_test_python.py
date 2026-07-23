"""
Workshop Step 4: Python Inference & Qualitative Evaluation
----------------------------------------------------------
This module loads the base model with the fine-tuned LoRA adapter 
and runs generation in Python using Hugging Face Transformers.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_DIR = "./LFM2.5-230M"
LORA_DIR = "./essay_reviewer_lora"

SAMPLE_ESSAY = """The recent push toward standardized testing has generated a great deal of debate in educational circles. While proponents tout the benefits of consistent metrics for measuring student progress, opponents argue that this approach stifles creativity and critical thinking. This essay will explore the multifaceted nature of standardized testing, arguing that while the metric provides an efficient means of assessing baseline knowledge, its implementation must be balanced with other evaluative methods to ensure a holistic understanding of student capabilities.
One of the most compelling arguments in favor of standardized testing is its capacity to provide a uniform benchmark for academic achievement. In a system where educational standards can vary drastically between states and even districts, standardized tests offer a reliable method of comparing student performance on a large scale. This consistency is particularly valuable for identifying trends in educational attainment and pinpointing areas where schools or students may be falling short.
However, the benefits of standardized testing are not without significant drawbacks. A primary concern is the potential for these tests to narrow the curriculum, forcing teachers to "teach to the test" rather than fostering a deeper understanding of the subject matter.
In conclusion, standardized testing offers a valuable, albeit limited, tool for measuring student achievement and ensuring educational consistency."""

def test_inference(essay_text=SAMPLE_ESSAY, base_model_dir=BASE_MODEL_DIR, lora_dir=LORA_DIR):
    print(f"[Step 4] Loading base model '{base_model_dir}' and LoRA adapter '{lora_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_dir)

    model = PeftModel.from_pretrained(base_model, lora_dir)
    print("[Step 4] Model & adapter loaded successfully!\n")

    message = [{"role": "user", "content": f"Review this essay and give feedback:\n\n{essay_text}"}]
    
    prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    print("[Step 4] Generating essay review feedback...")
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=256,
        do_sample=False,
    )

    feedback = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n--- MODEL FEEDBACK ---")
    print(feedback)
    print("----------------------\n")
    return feedback

if __name__ == "__main__":
    test_inference()
