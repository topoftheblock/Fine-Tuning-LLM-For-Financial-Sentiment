import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlx_lm import load, generate

app = FastAPI(title="M4 Financial Sentiment API (CoT + Multi-Ticker)")

print("Loading fused Qwen 0.5B model into M4 Unified Memory...")
MODEL_PATH = "./fused-qwen-finance" 
model, tokenizer = load(MODEL_PATH)
print("✅ Qwen Model loaded and API is ready!")

class AnalysisRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(request: AnalysisRequest):
    system_instruction = (
        "Analyze the following text. First, provide step-by-step reasoning inside <think>...</think> tags. "
        "Then, extract the tickers, sentiments (-1.0 to 1.0), and reasoning for ALL companies mentioned "
        "as a strict JSON array of objects."
    )
    
    prompt = (
        f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
        f"<|im_start|>user\nInput: {request.text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    try:
        # Increase max_tokens because CoT reasoning takes up more space!
        response_text = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=300, 
            verbose=False
        )
        
        # 1. Extract the Chain of Thought
        think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        thought_process = think_match.group(1).strip() if think_match else "No reasoning generated."
        
        # 2. Extract the JSON Array
        json_str = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        
        clean_json_array = json.loads(json_str)
        
        return {
            "chain_of_thought": thought_process,
            "signals": clean_json_array
        }

    except json.JSONDecodeError:
        return {
            "chain_of_thought": thought_process if 'thought_process' in locals() else "Failed.",
            "signals": [],
            "error": f"LLM Format Error. Raw output: {json_str if 'json_str' in locals() else response_text}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))