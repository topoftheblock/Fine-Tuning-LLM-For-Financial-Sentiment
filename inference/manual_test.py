import json
import re
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL_PATH = "./fused-qwen-finance" 

print(f" Loading CoT fine-tuned model from {MODEL_PATH} into M4 Unified Memory...")
try:
    model, tokenizer = load(MODEL_PATH)
    print(" Model loaded successfully!\n")
except Exception as e:
    print(f" Failed to load model. Error: {e}")
    exit(1)

def format_prompt(raw_text: str) -> str:
    system_instruction = (
        "Analyze the following text. First, provide step-by-step reasoning inside <think>...</think> tags. "
        "Then, extract the tickers, sentiments (-1.0 to 1.0), and reasoning for ALL companies mentioned "
        "as a strict JSON array of objects."
    )
    
    prompt = (
        f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
        f"<|im_start|>user\nInput: {raw_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt

print("="*60)
print("📈 FINANCIAL SENTIMENT ENGINE - CoT MULTI-TICKER")
print("Type 'exit' or 'quit' to stop.")
print("="*60)

strict_sampler = make_sampler(temp=0.1)

while True:
    user_input = input("\nEnter a financial tweet or headline:\n> ")
    
    if user_input.lower() in ['exit', 'quit']:
        break
    if not user_input.strip():
        continue
        
    formatted_prompt = format_prompt(user_input)
    print(" Analyzing (Thinking...)\n")
    
    response = generate(
        model, tokenizer, prompt=formatted_prompt, max_tokens=300, sampler=strict_sampler, verbose=False
    )
    
    # Parse the CoT block
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if think_match:
        print("\033[90m" + "🤔 AI THOUGHT PROCESS:" + "\033[0m") # Gray color for thoughts
        print("\033[90m" + think_match.group(1).strip() + "\033[0m\n")
    
    # Parse the JSON Array
    json_str = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    
    try:
        clean_json = json.loads(json_str)
        print("📊 EXTRACTED SIGNALS:")
        print(json.dumps(clean_json, indent=4))
    except json.JSONDecodeError:
        print("❌ FORMAT ERROR: Model failed to output valid JSON.")
        print(json_str)
        
    print("-" * 60)