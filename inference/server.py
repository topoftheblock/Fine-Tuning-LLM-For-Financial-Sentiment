import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlx_lm import load, generate

# --- 1. Initialize API and Load the Model ---
app = FastAPI(title="M4 Financial Sentiment API")

print("Loading fused Qwen 4B model into M4 Unified Memory...")
# Point this to the output directory where you fused the Qwen model
MODEL_PATH = "./fused-finance-model" 
model, tokenizer = load(MODEL_PATH)
print("✅ Qwen Model loaded and API is ready!")

# --- 2. Define the expected incoming data payload ---
class AnalysisRequest(BaseModel):
    text: str

# --- 3. Create the Inference Endpoint ---
@app.post("/analyze")
def analyze_text(request: AnalysisRequest):
    # Match the prompt EXACTLY to your training data format
    # No system prompt, just the user message
    prompt = (
        f"<|im_start|>user\n{request.text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    try:
        # Run inference using M4 MLX
        response_text = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=50, # Reduced because the expected JSON output is small
            verbose=False
        )
        
        # Clean the output and parse it back into a Python Dictionary (JSON)
        clean_json = json.loads(response_text.strip())
        
        # Ensure we always return the structure your Spark streaming script expects!
        # Since your training data doesn't have "reasoning", we provide a default string here.
        return {
            "ticker": clean_json.get("ticker", "UNKNOWN"),
            "sentiment": clean_json.get("sentiment", 0.0),
            "reasoning": "Not included in this model version."
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