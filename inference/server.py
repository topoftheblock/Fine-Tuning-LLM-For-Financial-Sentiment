import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlx_lm import load, generate

# 1. Initialize API and Load the Model
app = FastAPI(title="M4 Financial Sentiment API")

print("Loading fused Llama-3 model into M4 Unified Memory...")
# Point this to the output directory from your fuse_model.sh script
MODEL_PATH = "./fused-finance-model" 
model, tokenizer = load(MODEL_PATH)
print("✅ Model loaded and API is ready!")

# 2. Define the expected incoming data payload
class AnalysisRequest(BaseModel):
    text: str

# 3. Create the Inference Endpoint
@app.post("/analyze")
def analyze_text(request: AnalysisRequest):
    # Strictly format the prompt to match our Phase 2 training data
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Analyze the following text, extract the ticker, determine the sentiment "
        "(-1.0 to 1.0), and provide a brief reasoning. Output strictly in JSON format.\n\n"
        f"Input: {request.text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
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
        # We assume the model outputs clean JSON because we fine-tuned it to do so!
        clean_json = json.loads(response_text.strip())
        return clean_json

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