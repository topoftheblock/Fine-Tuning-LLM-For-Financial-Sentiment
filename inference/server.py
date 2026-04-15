import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlx_lm import load, generate

# --- 1. Initialize API and Load the Model ---
app = FastAPI(title="M4 Financial Sentiment API")

print("Loading fused Qwen 0.5B model into M4 Unified Memory...")
# UPDATED: Pointing to the correct fused Qwen model
MODEL_PATH = "./fused-qwen-finance" 
model, tokenizer = load(MODEL_PATH)
print("Qwen Model loaded and API is ready!")

# --- 2. Define the expected incoming data payload ---
class AnalysisRequest(BaseModel):
    text: str

# --- 3. Create the Inference Endpoint ---
@app.post("/analyze")
def analyze_text(request: AnalysisRequest):
    system_instruction = (
        "Analyze the following text, extract the ticker, determine the sentiment "
        "(-1.0 to 1.0), and provide a brief reasoning. Output strictly in JSON format."
    )
    
    # UPDATED: Matching the exact ChatML format used in the new distillation script
    prompt = (
        f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
        f"<|im_start|>user\nInput: {request.text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    try:
        # Run inference using M4 MLX
        response_text = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=150, 
            verbose=False
        )
        
        # Clean the output and parse it back into a Python Dictionary (JSON)
        clean_json = json.loads(response_text.strip())
        
        # Return the structure your Spark streaming script expects
        return {
            "ticker": clean_json.get("ticker", "UNKNOWN"),
            "sentiment": clean_json.get("sentiment", 0.0),
            "reasoning": clean_json.get("reasoning", "No reasoning provided.")
        }

    except json.JSONDecodeError:
        # Fallback in case the LLM hallucinates non-JSON text
        return {
            "ticker": "UNKNOWN", 
            "sentiment": 0.0, 
            "reasoning": f"LLM Format Error. Raw output: {response_text}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally using: uvicorn server:app --host 0.0.0.0 --port 8000