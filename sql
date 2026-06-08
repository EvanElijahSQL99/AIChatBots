from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import warnings
warnings.filterwarnings("ignore")

model_name = "defog/sqlcoder-7b-2"

# 4-bit quantization (QLoRA-style) so the 7B model fits on a smaller GPU (~6-8GB VRAM).
# Requires: pip install bitsandbytes accelerate
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model... (first run downloads several GB)")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,   # remove this line + use torch_dtype for full precision
    device_map="auto",
)

# --- Your database schema (edit this to match your tables) ---
schema = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    signup_date DATE,
    country TEXT
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    amount NUMERIC,
    order_date DATE
);
"""

# SQLCoder's expected prompt format
PROMPT_TEMPLATE = """### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""

print("SQLCoder ready! Ask a question in plain English. Type 'exit' to quit.\n")
while True:
    question = input("> ")
    if question.lower() == "exit":
        break

    prompt = PROMPT_TEMPLATE.format(question=question, schema=schema)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,            # deterministic — important for SQL
            num_beams=4,                # beam search improves SQL quality
            pad_token_id=tokenizer.eos_token_id,
        )

    # Keep only the newly generated tokens, then take the first SQL statement
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    sql = generated.split(";")[0].strip() + ";"
    print(f"\nSQL:\n{sql}\n")
