import os
import zipfile
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# ---------- CONFIG ----------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Use raw strings for Windows paths
DATA_FILE  = r"C:\Users\HP\Desktop\Analyst\train_data_cleaned.jsonl"
OUTPUT_DIR = r"Z:\model\finetuned-qwen2.5-3b"
HF_CACHE   = r"Z:\model\hf_cache"

# ---------- SETUP ----------

os.environ["HF_HOME"] = HF_CACHE
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

has_cuda = torch.cuda.is_available()
device = "cuda" if has_cuda else "cpu"

print("CUDA available:", has_cuda)
print("Device:", device)
if has_cuda:
    print("GPU:", torch.cuda.get_device_name(0))

# ---------- LOAD TOKENIZER & MODEL ----------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# On CPU: use float32. On GPU: you can switch to bfloat16 if supported.
dtype = torch.bfloat16 if has_cuda else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    cache_dir=HF_CACHE,   # no device_map here
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# Move model to device (CPU or GPU)
model.to(device)

# ---------- LoRA CONFIG ----------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------- DATASET PREP ----------

dataset = load_dataset("json", data_files=DATA_FILE)

def format_example(example):
    # adjust keys if your JSONL uses different names
    return f"### Prompt:\n{example['prompt']}\n\n### Response:\n{example['completion']}"

def tokenize(example):
    text = format_example(example)
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=1024,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset["train"].map(
    tokenize,
    batched=False,
    remove_columns=dataset["train"].column_names
)

# ---------- TRAINING ARGS ----------

use_bf16 = has_cuda and torch.cuda.is_bf16_supported()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=50,          # ↓ start small on CPU; you can increase later
    warmup_steps=5,
    logging_steps=5,
    bf16=use_bf16,
    fp16=False,            # keep False on CPU
    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=1024,  # deprecation warning but still works
)

trainer.train()

# ---------- SAVE MODEL & TOKENIZER ----------

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ---------- ZIP THE OUTPUT DIR LOCALLY ----------

zip_path = os.path.join(os.path.dirname(OUTPUT_DIR), "finetuned-qwen2.5-3b.zip")

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, path)
            ziph.write(full_path, rel_path)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    zipdir(OUTPUT_DIR, zipf)

print("Training complete.")
print(f"Model dir: {OUTPUT_DIR}")
print(f"Zipped model: {zip_path}")
