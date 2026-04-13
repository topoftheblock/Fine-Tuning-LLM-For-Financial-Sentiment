import json
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler  # <-- NEW: Import the sampler utility

# --- 1. Configuration & Loading ---
MODEL_PATH = "./fused-finance-model" 

print(f"🧠 Loading fine-tuned model from {MODEL_PATH} into M4 Unified Memory...")
print("Please wait, this usually takes 5-10 seconds...")
try:
    model, tokenizer = load(MODEL_PATH)
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Failed to load model. Did you run the fusion script? Error: {e}")
    exit(1)

# --- 2. The Strict Prompt Template ---
def format_prompt(raw_text: str) -> str:
    """Wraps incoming text in the exact Llama-3 format used during training."""
    
    system_instruction = (
        "Analyze the following text, extract the ticker, determine the sentiment "
        "(-1.0 to 1.0), and provide a brief reasoning. Output strictly in JSON format."
    )
    
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{system_instruction}\n\n"
        f"Input: {raw_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    return prompt

# --- 3. Interactive Testing Loop ---
print("="*60)
print("📈 FINANCIAL SENTIMENT ENGINE - MANUAL OVERRIDE")
print("Type 'exit' or 'quit' to stop.")
print("="*60)

# NEW: Create a deterministic sampler to enforce strict JSON formatting
strict_sampler = make_sampler(temp=0.0)

while True:
    user_input = input("\nEnter a financial tweet or headline:\n> ")
    
    if user_input.lower() in ['exit', 'quit']:
        print("Shutting down engine...")
        break
        
    if not user_input.strip():
        continue
        
    formatted_prompt = format_prompt(user_input)
    
    print("🤖 Analyzing...")
    
    # Generate the response using our strict sampler
    response = generate(
        model, 
        tokenizer, 
        prompt=formatted_prompt, 
        max_tokens=150, 
        sampler=strict_sampler,  # <-- FIX: Pass the sampler object here
        verbose=False
    )
    
    # Clean up any trailing spaces the model might have added
    raw_output = response.strip()
    
    # --- 4. Validation & Output ---
    try:
        clean_json = json.loads(raw_output)
        
        print("\n✅ SUCCESSFUL EXTRACTION:")
        print(json.dumps(clean_json, indent=4))
        
    except json.JSONDecodeError:
        print("\n⚠️ FORMAT ERROR: Model failed to output valid JSON.")
        print("Raw output from model:")
        print(raw_output)
        
    print("-" * 60)