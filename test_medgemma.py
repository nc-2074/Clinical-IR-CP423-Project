from mlx_lm import load, generate
from dotenv import load_dotenv
import os

load_dotenv()

print("Loading MedGemma 4B via MLX...")
model, tokenizer = load(
    "mlx-community/medgemma-4b-it-4bit",
    tokenizer_config={"trust_remote_code": True},
)
print("Model loaded!")

messages = [
    {
        "role": "system",
        "content": "You are a clinical documentation assistant. Summarize clinical interviews accurately and concisely. This is for educational purposes only."
    },
    {
        "role": "user", 
        "content": "A patient says they have a headache for two days and no fever. The clinician prescribed medicine after meals. Summarize this."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=200,
    verbose=True
)

print(response)